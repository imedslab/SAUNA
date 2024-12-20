import torch
import torch.nn as nn
import torch.nn.functional as functional
import monai.losses
from einops import rearrange
from torch.nn.modules.loss import _Loss

from segmentation_models_pytorch.losses import JaccardLoss, DiceLoss, TverskyLoss, FocalLoss, LovaszLoss
from segmentation_models_pytorch.losses import SoftBCEWithLogitsLoss
from mlpipeline.losses.boundary_loss import BoundaryLoss, class2one_hot, simplex


class HingeLoss(_Loss):
    def __init__(self, eps: float = 1.0, **kwargs):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        bs = y_true.shape[0]
        y_pred = torch.tanh(y_pred)
        y_true = y_true.view(bs, -1)
        y_pred = y_pred.view(bs, -1)

        diff = self.eps - y_pred * y_true

        mask = diff > 0

        count = torch.sum(mask)

        loss = torch.mean(diff[mask])

        # loss = torch.sum(torch.log(1 + torch.exp(- y_pred * y_true))) / count

        return loss


class BinaryFocalLoss(nn.Module):
    def __init__(self):
        super(BinaryFocalLoss, self).__init__()

    @staticmethod
    def binary_focal(pred, gt, gamma=2, *args):
        return -gt * torch.log(pred) * torch.pow(1 - pred, gamma)

    def forward(self, pred, gt, gamma=2, eps=1e-6, *args):
        pred = torch.clamp(pred, eps, 1 - eps)
        loss1 = self.binary_focal(pred, gt, gamma=gamma)
        loss2 = self.binary_focal(1 - pred, 1 - gt, gamma=gamma)
        loss = loss1 + loss2
        return loss.mean()


class GeneralizedDice():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs, target):
        assert simplex(probs) and simplex(target)

        pc = probs[:, self.idc, ...].type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)

        w = 1 / ((torch.einsum("bkwh->bk", tc).type(torch.float32) + 1e-10) ** 2)
        intersection = w * torch.einsum("bkwh,bkwh->bk", pc, tc)
        union = w * (torch.einsum("bkwh->bk", pc) +
                     torch.einsum("bkwh->bk", tc))

        divided = 1 - 2 * (torch.einsum("bk->b", intersection) +
                           1e-10) / (torch.einsum("bk->b", union) + 1e-10)

        loss = divided.mean()
        return loss


class GDI_BL(nn.Module):
    def __init__(self):
        super(GDI_BL, self).__init__()
        self.alpha = 0.01
        self.epoch = 0

        self.criterion_gdi = GeneralizedDice(idc=[0, 1])
        self.criterion_bl = BoundaryLoss(idc=[1])

    def update_alpha(self, epoch=None):
        if epoch is None:
            self.epoch += 1
        else:
            self.epoch = epoch

        if (self.epoch > 50) and (self.epoch % 1 == 0):
            self.alpha += 0.01
        if self.alpha >= 1.0:
            self.alpha = 0.01
        return self.alpha

    def forward(self, pred, gt, dist_map):
        assert pred.shape[1] == 2

        probs = functional.softmax(pred, dim=1)
        gt = class2one_hot(gt, K=2)

        loss_gdi = self.criterion_gdi(probs, gt)
        loss_bl = self.criterion_bl(probs, dist_map)
        loss = (1.0 - self.alpha) * loss_gdi + self.alpha * loss_bl
        # loss = loss_gdi

        # self.update_alpha()
        return loss


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, pred, gt):
        loss = self.criterion(pred, gt.squeeze(dim=1).long())
        return loss


class DiceBCE(nn.Module):
    def __init__(self, mode, from_logits, smooth, pos_weight):
        super(DiceBCE, self).__init__()
        self.dice = DiceLoss(mode=mode, from_logits=from_logits, smooth=smooth)
        self.bce = SoftBCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]), smooth_factor=smooth)

    def forward(self, pred, gt):
        loss = self.dice(pred, gt)
        loss = loss + self.bce(pred, gt.float())
        return loss


class SFLoss(nn.Module):
    def __init__(self, loss_type=1, alpha=1.0, beta=1.0, secondary_weight=1.0, pos_weight=1.0, base_loss="l1"):
        super(SFLoss, self).__init__()
        self.base_loss = base_loss

        if base_loss == "smoothl1":
            self.criterion = nn.SmoothL1Loss(reduction="none")
        elif base_loss == "l2":
            self.criterion = nn.MSELoss(reduction="none")
        else:
            self.criterion = nn.L1Loss(reduction="none")

        self.loss_type = loss_type
        self.alpha = alpha
        self.beta = beta
        self.pos_weight = pos_weight
        self.secondary_weight = secondary_weight

    def forward(self, pred, gt):
        same_sign_mask = (torch.sign(pred) * torch.sign(gt) > 0)
        pos_mask = (gt > 0)
        weight = torch.ones_like(gt)

        weight = torch.where(
            same_sign_mask,
            torch.pow(torch.abs(pred - gt), self.beta) * self.alpha,
            torch.ones_like(gt),
        )
        weight = weight * torch.where(
            pos_mask,
            torch.full_like(gt, fill_value=self.pos_weight),
            torch.ones_like(gt),
        )

        if self.loss_type > 1:
            raise ValueError("Invalid type!")
        if self.loss_type in [0]:
            weight = weight.detach()

        loss = self.criterion(pred, gt)
        loss = loss * weight
        loss = torch.mean(loss, dim=(0, 2, 3))

        class_weights = torch.tensor(
            [1.0] + [self.secondary_weight] * (loss.shape[0] - 1), device=loss.device)
        loss = (loss * class_weights).sum() / class_weights.sum()
        return loss


class DistanceLoss(nn.Module):
    def __init__(self, name, **kwargs):
        super(DistanceLoss, self).__init__()
        self.distance_name = name.replace("distance-", "").lower()
        loss_type = kwargs.get("loss_type", 1)
        alpha = kwargs.get("alpha", 2.0)
        beta = kwargs.get("beta", 1.0)
        pos_weight = kwargs.get("pos_weight", 1.0)
        secondary_weight = kwargs.get("secondary_weight", 1.0)
        base_loss = kwargs.get("base_loss", "sfl1")

        self.criterion = {
            "l2": nn.MSELoss(),
            "l1": nn.L1Loss(),
            "smoothl1": nn.SmoothL1Loss(),
            "sfl1": SFLoss(
                loss_type=loss_type,
                alpha=alpha,
                beta=beta,
                pos_weight=pos_weight,
                secondary_weight=secondary_weight,
                base_loss=base_loss),
        }[self.distance_name]

    def forward(self, pred, gt):
        gt = gt[:, 1:, :, :]
        loss = self.criterion(torch.tanh(pred), gt)
        return loss


class ProductLoss(nn.Module):
    def __init__(self):
        super(ProductLoss, self).__init__()
        self.smooth = 1e-6
        self.criterion = nn.L1Loss()

    def forward(self, pred, gt):
        gt = gt[:, 1:, :, :]
        pred = torch.tanh(pred)
        assert gt.shape[1] == pred.shape[1] == 1

        pred = pred.view(pred.shape[0], -1)
        gt = gt.view(gt.shape[0], -1)

        pt2 = torch.sum(pred * pred, dim=-1)
        yt2 = torch.sum(gt * gt, dim=-1)
        ytpt = torch.sum(gt * pred, dim=-1)

        loss = (ytpt + self.smooth) / (ytpt + pt2 + yt2 + self.smooth)
        loss = -torch.mean(loss)
        loss = loss + self.criterion(pred, gt)
        return loss


class SoftJaccardLoss(nn.Module):
    def __init__(self, loss_type=1, **kwargs):
        super(SoftJaccardLoss, self).__init__()
        self.smooth = 1e-8
        self.loss_type = loss_type
        self.secondary_weight = kwargs.get("secondary_weight", 1.0)

    def compute_iou(self, x, y):
        norm_sum = torch.sum(torch.abs(x + y))
        norm_dif = torch.sum(torch.abs(x - y))

        inter = (norm_sum - norm_dif)
        union = (norm_sum + norm_dif)
        iou = inter / (union + self.smooth)
        return iou, inter, union

    def _compute_positive_jaccard(self, prob, label):
        cardinality = torch.sum(prob + label, dim=1)
        difference = torch.sum(torch.abs(prob - label), dim=1)
        tp = (cardinality - difference) / 2
        fp = torch.sum(prob, dim=1) - tp
        fn = torch.sum(label, dim=1) - tp
        tp = torch.sum(tp, dim=0)
        fp = torch.sum(fp, dim=0)
        fn = torch.sum(fn, dim=0)
        tversky = (tp + self.smooth) / (tp + fp + fn + self.smooth)
        loss = torch.mean(1.0 - tversky)
        return loss

    def _compute_loss_on_class(self, pred, gt, index):
        gt = gt[:, index:(index + 1), :, :]
        pred = pred[:, (index - 1):index, :, :]
        assert gt.shape[1] == pred.shape[1] == 1
        gt = gt.view(gt.shape[0], -1)
        pred = pred.view(pred.shape[0], -1)

        if gt.min() >= 0.0:
            pred = torch.sigmoid(pred)
            loss = self._compute_positive_jaccard(pred, gt)
            return loss

        pred = torch.tanh(pred)
        fg_iou, fg_i, fg_u = self.compute_iou(torch.relu(pred), torch.relu(gt))
        bg_iou, bg_i, bg_u = self.compute_iou(
            torch.relu(-pred), torch.relu(-gt))

        if self.loss_type == 1:
            fg_loss = 1.0 - fg_iou
            bg_loss = 1.0 - bg_iou
            loss = fg_loss + bg_loss
        elif self.loss_type == 2:
            loss = 1.0 - (fg_i + bg_i) / (fg_u + bg_u)
        else:
            raise ValueError("Invalid type!")

        return loss

    def forward(self, pred, gt):
        loss = self._compute_loss_on_class(pred, gt, index=1)

        for i in range(2, gt.shape[1]):
            loss += self._compute_loss_on_class(pred,
                                                gt, index=i) * self.secondary_weight

        return loss


class DistanceSoftJaccardLoss(nn.Module):
    def __init__(self, name, **kwargs):
        super(DistanceSoftJaccardLoss, self).__init__()

        distance_name = name.replace("distancesjm-", "distance-").lower()
        self.distance_loss = DistanceLoss(distance_name, **kwargs)
        self.sjm_loss = SoftJaccardLoss(loss_type=2)
        self.distance_weight = kwargs["distance_weight"]

    def forward(self, pred, gt):
        distance_loss = self.distance_loss(pred, gt)
        sjm_loss = self.sjm_loss(pred, gt)
        # print("Losses:", distance_loss, sjm_loss)

        loss = (
            distance_loss
            +
            self.distance_weight * sjm_loss
        )
        return loss


def create_segmentation_loss(name, **kwargs):
    mode = kwargs["mode"]
    from_logits = kwargs["from_logits"]
    smooth = kwargs["smooth"]
    pos_weight = kwargs.get("pos_weight", 1.0)

    if name == "dice":
        return DiceLoss(mode=mode, from_logits=from_logits, smooth=smooth)
    elif name == "jaccard":
        return JaccardLoss(mode=mode, from_logits=from_logits, smooth=smooth)
    elif name == "tversky":
        return TverskyLoss(mode=mode, from_logits=from_logits, smooth=smooth)
    elif name == "focal":
        return FocalLoss(**kwargs)
    elif name == "binary-focal":
        return BinaryFocalLoss()
    elif name == "lovasz":
        return LovaszLoss(**kwargs)
    elif name == "bce":
        return SoftBCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    elif name == "dicebce":
        return DiceBCE(mode=mode, from_logits=from_logits, smooth=smooth, pos_weight=pos_weight)
    elif name == "hinge":
        return HingeLoss(**kwargs)
    elif name == "ce":
        return CrossEntropyLoss()
    elif name == "gdi-bl":
        return GDI_BL()
    elif "distance-" in name:
        return DistanceLoss(name, **kwargs)
    elif "distancesjm-" in name:
        return DistanceSoftJaccardLoss(name, **kwargs)
    elif name == "product":
        return ProductLoss()
    elif name == "sjm":
        return SoftJaccardLoss()
    else:
        raise ValueError(f'Not support loss {name}.')
