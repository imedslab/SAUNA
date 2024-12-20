import types

import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import monai
from torch import nn
from torchvision import transforms as T
from segmentation_models_pytorch.encoders import encoders

from mlpipeline.losses import create_segmentation_loss
from mlpipeline.models.segmentation.utils import get_stages
from mlpipeline.models.acc_unet import ACC_UNet
from mlpipeline.models.segmentation.r2unet import R2UNet
from mlpipeline.models.segmentation.iternet import IterNet
from mlpipeline.models.segmentation.ctf_net import LadderNetv6
from mlpipeline.models.segmentation.cenet.cenet import CE_Net_
from mlpipeline.models.segmentation.cga_net import CGAM_UNet2
from mlpipeline.models.segmentation.dunet.deform_unet import DUNetV1V2
from mlpipeline.models.segmentation.fr_unet import FR_UNet
from mlpipeline.models.segmentation.sa_unet.sa_unet import SA_Unet
from mlpipeline.models.segmentation.scsnet.scsnet import SCSNet
from mlpipeline.models.segmentation.magfnet.magfnet import MagfNet
from mlpipeline.models.segmentation.skelcon.model_siam import build_model, MLPSampler
from mlpipeline.models.segmentation.danet.danet import DANet
from mlpipeline.models.segmentation.swin_unet.vision_transformer import SwinUnet
from mlpipeline.models.segmentation.sgat.sgat import SGAT


class SemanticSegmentation(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.create_criterion()
        self.model = self.get_model()

    def get_model(self):
        if self.cfg.arch in ["Unet", "UnetPlusPlus", "DeepLabV3Plus"]:
            self.model = smp.create_model(
                self.cfg.arch,
                encoder_name=self.cfg.encoder_name,
                encoder_weights=self.cfg.encoder_weights,
                in_channels=self.cfg.num_channels,
                classes=self.cfg.num_classes,
            )

        elif self.cfg.arch == "ACC-Unet":
            self.model = ACC_UNet(
                n_channels=self.cfg.num_channels,
                n_classes=self.cfg.num_classes,
                n_filts=8,
            )

        elif self.cfg.arch == "R2Unet":
            self.model = R2UNet(
                num_channels=self.cfg.num_channels,
                num_classes=self.cfg.num_classes,
                num_filters=16,
            )

        elif self.cfg.arch == "IterNet":
            self.model = IterNet(
                num_channels=self.cfg.num_channels,
                num_classes=self.cfg.num_classes,
                num_filters=32,
                dropout=self.cfg.dropout,
                activation="relu",
                iteration=3,
            )

        elif self.cfg.arch == "CTF-Net":
            self.model = LadderNetv6(
                inplanes=self.cfg.num_channels,
                num_classes=self.cfg.num_classes,
                layers=4,
                filters=16,
            )

        elif self.cfg.arch == "CE-Net":
            self.model = CE_Net_(
                num_channels=self.cfg.num_channels,
                num_classes=self.cfg.num_classes,
            )

        elif self.cfg.arch == "CGA-Net":
            self.model = CGAM_UNet2(
                n_channels=self.cfg.num_channels,
                n_classes=self.cfg.num_classes,
            )

        elif self.cfg.arch == "DUnet":
            self.model = DUNetV1V2(
                n_channels=self.cfg.num_channels,
                n_classes=self.cfg.num_classes,
                downsize_nb_filters_factor=4,
            )

        elif self.cfg.arch == "FR-Unet":
            self.model = FR_UNet(
                num_channels=self.cfg.num_channels,
                num_classes=self.cfg.num_classes,
                fuse=True, out_average=True,
                dropout=self.cfg.dropout,
            )

        elif self.cfg.arch == "SA-Unet":
            self.model = SA_Unet(
                num_channels=self.cfg.num_channels,
                num_classes=self.cfg.num_classes,
                start_neurons=16,
            )

        elif self.cfg.arch in ["Unet_modified", "UnetPlusPlus_modified"]:
            encoders["resnet50"]["params"]["out_channels"] = (
                64, 256, 512, 1024, 2048)
            self.model = smp.create_model(
                self.cfg.arch.replace("_modified", ""),
                encoder_name=self.cfg.encoder_name,
                encoder_weights=self.cfg.encoder_weights,
                in_channels=self.cfg.num_channels,
                classes=self.cfg.num_classes,
                encoder_depth=4,
                decoder_channels=(256, 128, 64, 32),
            )
            if "resnet50" in self.cfg.encoder_name:
                self.model.encoder.conv1.stride = (1, 1)
                self.model.encoder.get_stages = types.MethodType(
                    get_stages, self.model.encoder)

        elif self.cfg.arch == "SCSNet":
            self.model = SCSNet(
                in_channels=self.cfg.num_channels,
                num_classes=self.cfg.num_classes,
                super_resolution=False,
                upscale_rate=1,
            )

        elif self.cfg.arch == "MAGF-Net":
            self.model = MagfNet(
                in_channels=self.cfg.num_channels,
                num_classes=self.cfg.num_classes,
                features_dim=64,
            )

        elif self.cfg.arch == "SkelCon":
            self.model = build_model(
                in_channels=self.cfg.num_channels,
                type_net="lunet",
                type_seg="lunet",
                type_loss="sim2",
                num_emb=128,
            )
            self.sampler = MLPSampler(mode="hard", select3=False)
            self.criterion.from_logits = False

        elif self.cfg.arch == "DA-Net":
            self.model = DANet(
                in_channels=self.cfg.num_channels,
                out_channels=self.cfg.num_classes,
                dropout=self.cfg.dropout,
                num_patches=self.cfg.two_branches_num_patches,
            )

        elif self.cfg.arch == "Swin-Unet":
            self.model = SwinUnet(
                config=None,
                new_config=self.cfg,
                image_size=self.cfg.image_size,
                num_classes=self.cfg.num_classes,
            )

        elif self.cfg.arch == "SGAT":
            self.model = SGAT(
                in_channels=self.cfg.num_channels,
                num_classes=self.cfg.num_classes,
                base_width=32,
                num_patches=16 * 16,
                dropout=self.cfg.dropout,
            )

        else:
            raise ValueError("Invalid architecture!")

        return self.model

    def get_rank(self):
        return next(self.parameters()).device

    def create_criterion(self):
        from_logits = False if self.cfg.arch in ["SkelCon"] else True
        distance_weight = 0.5 if self.cfg.distance_weight is None else self.cfg.distance_weight
        loss_type = 1 if self.cfg.loss_type is None else self.cfg.loss_type

        self.criterion = create_segmentation_loss(
            self.cfg.loss_name,
            mode=self.cfg.loss_mode,
            from_logits=from_logits,
            smooth=0,
            loss_type=loss_type,
            alpha=self.cfg.alpha,
            beta=self.cfg.beta,
            pos_weight=self.cfg.pos_weight,
            secondary_weight=self.cfg.secondary_weight,
            base_loss=self.cfg.base_loss,
            distance_weight=distance_weight,
        )

    def forward(self, batch):
        img_input = batch["input"].cuda(self.get_rank(), non_blocking=True)
        img_gt = batch["gt"].cuda(self.get_rank(), non_blocking=True)

        # print(img_input.amin(dim=(0, 2, 3)), img_input.amax(dim=(0, 2, 3)), img_gt.min(), img_gt.max())
        pred = self.model(img_input)
        assert img_input.ndim >= 4
        assert img_gt.ndim == 4 or self.cfg.loss_name == "gdi-bl"

        if self.cfg.loss_name == "gdi-bl":
            dist_map = batch["dist_map"].cuda(
                self.get_rank(), non_blocking=True)
            loss = self.criterion(pred, img_gt, dist_map)
            outputs = {"pred": pred, "gt": img_gt}
        elif self.cfg.loss_name == 'hinge':
            loss = self.criterion(pred, img_gt)
            img_gt = (img_gt >= 0) * 1.0
            outputs = {"pred": pred, "gt": img_gt}
        elif self.cfg.arch == "SkelCon":
            skeleton = batch["skeleton"].cuda(
                self.get_rank(), non_blocking=True)
            fov_mask = None

            loss_seg = self.criterion(
                pred[0], img_gt) + 0.9 * self.criterion(pred[1], img_gt)
            loss_scl = self.model.regular(
                self.sampler, lab=img_gt, fov=fov_mask)
            los1, los2 = self.model.constraint(
                lab=img_gt, aux=skeleton, fov=fov_mask)
            loss_sfd = (los1 * 0.5 + los2)

            loss = loss_seg + 0.1 * loss_scl + 0.9 * loss_sfd
            outputs = {"pred": pred[0], "gt": img_gt}
        elif isinstance(pred, list) or isinstance(pred, tuple):
            # IterNet, SCSNet
            loss = sum([self.criterion(out, img_gt) for out in pred])
            outputs = {"pred": pred[-1], "gt": img_gt}
        else:
            loss = self.criterion(pred, img_gt)
            outputs = {"pred": pred, "gt": img_gt}

        return {"loss": loss}, outputs

    def predict(self, batch, inferer=None):
        img_input = batch["input"].cuda(self.get_rank(), non_blocking=True)
        img_gt = batch["gt"].cuda(self.get_rank(), non_blocking=True)

        if self.cfg.test_on_patches and (inferer is not None):
            # SlidingWindowInferer will forcefully get the first output in tuple
            pred = inferer(inputs=img_input, network=self.model)
        else:
            pred = self.model(img_input)

        if self.cfg.arch == "SkelCon":
            pred = pred[0]
        elif isinstance(pred, list) or isinstance(pred, tuple):
            # IterNet, SCSNet
            pred = pred[-1]

        outputs = {"pred": pred, "gt": img_gt}
        return outputs
