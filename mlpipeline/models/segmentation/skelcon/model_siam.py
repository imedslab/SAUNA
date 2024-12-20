import torch
import torch.nn as nn
import torch.nn.functional as functional

from mlpipeline.models.segmentation.skelcon.model_unet import lunet
from mlpipeline.models.segmentation.skelcon.sampler import MLPSampler


def similar_matrix2(q, k, temperature=0.1):
    # print("similar_matrix2:", q.shape, k.shape)
    qfh, qfl, qbh, qbl = torch.chunk(q, 4, dim=0)
    kfh, kfl, kbh, kbl = torch.chunk(k, 4, dim=0)

    # negative logits: NxK
    l_pos = torch.einsum("nc,kc -> nk", [qfl, kfh])
    l_neg = torch.einsum("nc,kc -> nk", [qbl, kbh])
    return 2.0 - l_pos.mean() - l_neg.mean()


CLLOSSES = {
    "sim2": similar_matrix2,
    "sim3": similar_matrix2,
}


class SIAM(nn.Module):
    __name__ = "siam"

    def __init__(
        self,
        encoder,
        clloss="nce",
        temperature=0.1,
        proj_num_layers=2,
        pred_num_layers=2,
        proj_num_length=64,
        **kwargs,
    ):
        super().__init__()
        self.loss = CLLOSSES[clloss]
        self.encoder = encoder
        self.__name__ = "X".join([self.__name__, self.encoder.__name__])
        # , clloss

        self.temperature = temperature
        self.proj_num_layers = proj_num_layers
        self.pred_num_layers = pred_num_layers

        self.projector = self.encoder.projector
        self.predictor = self.encoder.predictor

    def forward(self, image, **args):
        out = self.encoder(image, **args)
        self.pred = self.encoder.pred
        self.feat = self.encoder.feat
        # self._dequeue_and_enqueue(proj1_ng, proj2_ng)

        if hasattr(self.encoder, "tmp"):
            self.tmp = self.encoder.tmp
        return out

    def constraint(self, **args):
        return self.encoder.constraint(**args)

    def sphere(self, sampler, lab, fov=None):
        # contrastive loss split by classification
        # proj_num_length is used here
        feat = sampler.select(self.feat, self.pred.detach(), lab, fov)
        feat = self.projector(feat)
        self.proj = feat
        true_target = torch.zeros(
            size=(feat.shape[0],), dtype=torch.long).to(feat.device)
        true_target[:(feat.shape[0] // 2)] = 1
        return self.loss(feat, true_target)

    def regular(self, sampler, lab, fov=None):
        # contrastive loss split by classification
        feat = sampler.select(self.feat.clone(), self.pred.detach(), lab, fov)
        proj = self.projector(feat)
        self.proj = proj
        pred = self.predictor(proj)

        # compute loss
        losSG1 = self.loss(pred, proj.detach(), temperature=self.temperature)
        losSG2 = self.loss(proj, pred.detach(), temperature=self.temperature)
        return losSG1 + losSG2


class MlpNorm(nn.Module):
    def __init__(self, in_channels=256, out_channels=64):
        super(MlpNorm, self).__init__()
        features_dim = min(in_channels, out_channels)
        # max(dim_inp, dim_out) // 2

        # hidden layers
        linear_hidden = []
        linear_hidden.append(nn.Linear(in_channels, features_dim))
        linear_hidden.append(nn.Dropout(p=0.2))
        linear_hidden.append(nn.BatchNorm1d(features_dim))
        linear_hidden.append(nn.LeakyReLU())
        self.linear_hidden = nn.Sequential(*linear_hidden)

        self.linear_out = nn.Linear(features_dim, out_channels)
        # if num_layers >= 1 else nn.Identity()

    def forward(self, x):
        x = self.linear_hidden(x)
        x = self.linear_out(x)
        return functional.normalize(x, p=2, dim=-1)


def torch_dilation(x, kernel_size=3, stride=1):
    return functional.max_pool2d(x, (kernel_size, kernel_size), stride, kernel_size // 2)


class MorphBlock(nn.Module):
    def __init__(self, in_channels=2, channels=8):
        super().__init__()
        self.ch_wv = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=5, padding=2),
            nn.Conv2d(channels, channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1),
        )
        self.ch_wq = nn.Sequential(
            nn.Conv2d(channels//2, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x, o):
        x = torch.cat([torch_dilation(o, kernel_size=3), x, o], dim=1)
        # , 1-torch_dilation(1-x, ksize=3)
        x = self.ch_wv(x)
        return self.ch_wq(x)


class DiceLoss(nn.Module):
    __name__ = 'DiceLoss'
    # DSC(A, B) = 2 * |A ^ B | / ( | A|+|B|)

    def __init__(self, ):
        super(DiceLoss, self).__init__()
        self.func = self.dice

    def forward(self, pr, gt, **args):
        return 2 - self.dice(pr, gt) - self.dice(1-pr, 1-gt)
        # return 1-self.func(pr, gt)

    @staticmethod
    def dice(pr, gt, smooth=1):
        pr, gt = pr.view(-1), gt.view(-1)
        inter = (pr * gt).sum()
        union = (pr + gt).sum()
        return (smooth + 2*inter) / (smooth + union)


class SeqNet(nn.Module):
    # Supervised contrastive learning segmentation network
    __name__ = "scls"
    tmp = {}

    def __init__(self, in_channels, type_net, type_seg, num_emb=128):
        super(SeqNet, self).__init__()

        self.fcn = eval(
            f"{type_net}(in_channels={in_channels}, num_emb={num_emb})")
        self.seg = eval(f"{type_seg}(in_channels=32)")

        # self.fcn.projector#MlpNorm(32, 64, num_emb)
        self.projector = MlpNorm(32, num_emb)
        # self.fcn.predictor#MlpNorm(32, 64, num_emb)
        self.predictor = MlpNorm(num_emb, num_emb)

        self.morpholer1 = MorphBlock(32+2)
        self.morpholer2 = MorphBlock(32+2)

        self.__name__ = "{}X{}".format(self.fcn.__name__, self.seg.__name__)

    def constraint(self, aux=None, **args):
        aux = torch_dilation(aux)
        loss1 = DiceLoss.dice(self.sdm1, aux)
        loss2 = DiceLoss.dice(self.sdm2, aux)
        # if self.__name__.__contains__("dmf"):
        # 	loss1 = loss1 + self.fcn.regular()*0.1
        return loss1, loss2

    def regular(self, sampler, lab, fov=None, return_loss=True):
        emb = sampler.select(self.feat.clone(), self.pred.detach(), lab, fov)
        emb = self.projector(emb)
        self.emb = emb
        if return_loss:
            return sampler.infonce(emb)
        return

    def forward(self, x):
        aux = self.fcn(x)
        self.feat = self.fcn.feat
        out = self.seg(self.feat)
        self.pred = out

        self.sdm1 = self.morpholer1(self.fcn.feat, aux)
        self.sdm2 = self.morpholer2(self.seg.feat, out)
        self.tmp = {"sdm1": self.sdm1, "sdm2": self.sdm2}

        if self.training:
            if isinstance(aux, (tuple, list)):
                return [self.pred, aux[0], aux[1]]
            else:
                return [self.pred, aux]
        return [self.pred, aux]


def build_model(in_channels, type_net="lunet", type_seg="lunet", type_loss="sim2", num_emb=128):
    # model = lunet(num_emb=num_emb)

    if type_seg not in [""]:
        model = SeqNet(in_channels, type_net, type_seg, num_emb=num_emb)
        model = SIAM(encoder=model, clloss=type_loss, proj_num_length=num_emb)
    return model


def check():
    model = build_model(in_channels=1)
    model.train()
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    outs = model(torch.rand(2, 1, 128, 128))
    print([(out.shape, out.min(), out.max()) for out in outs])

    select3 = False
    sampler = MLPSampler(mode="hard", select3=select3)
    gt = torch.zeros(2, 1, 128, 128)
    gt[:, 0, 50:, 50:] = 1
    fov = torch.ones(2, 1, 128, 128) if select3 else None
    print(model.regular(sampler, lab=gt, fov=fov))


if __name__ == "__main__":
    check()
