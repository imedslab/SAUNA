import torch
import torch.nn as nn
import torch.nn.functional as functional

import mlpipeline.models.segmentation.scsnet.layers as layers
from mlpipeline.models.segmentation.scsnet.layers import Conv2d_BN


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True, activation=nn.ReLU()):
        """
            Implementation of the residual block in SCS-Net, we follow the structure
            described in ResNet to build this block.
            References:
                "SCS-Net: A Scale and Context Sensitive Network for Retinal Vessel Segmentation"

            Parameters:
            ----------
                in_ch
                out_ch
                norm_layer
                activation
        """
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            Conv2d_BN(in_channels, out_channels, kernel_size=3,
                      stride=1, padding=1, use_bn=use_bn),
            activation,
            Conv2d_BN(out_channels, out_channels, kernel_size=3,
                      stride=1, padding=1, use_bn=use_bn),
        )
        self.activation = activation

        self.identity = nn.Identity()
        if in_channels != out_channels:
            self.identity = Conv2d_BN(
                in_channels, out_channels, 1, 1, bias=False)
        return

    def forward(self, x):
        identity = self.identity(x)
        net = self.conv(x)
        net = net + identity
        net = self.activation(net)
        return net


class SCSNet(nn.Module):
    def __init__(
        self, in_channels=3, num_classes=2,
        super_resolution=False, output_size=None, upscale_rate=2,
        alphas=[0.6, 0.3, 0.1],
    ):
        super(SCSNet, self).__init__()
        base_channels = 64
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder1 = ResidualBlock(in_channels, base_channels)
        self.encoder2 = ResidualBlock(base_channels, base_channels * 2)
        self.encoder3 = ResidualBlock(base_channels * 2, base_channels * 4)

        self.sfa = layers.SFA(base_channels * 4)

        self.aff3 = layers.AFF(base_channels * 4)
        self.aff2 = layers.AFF(base_channels * 2)
        self.aff_conv3 = nn.Sequential(
            Conv2d_BN(base_channels * 4, base_channels * 2,
                      3, stride=1, padding=1, bias=False),
            nn.ReLU(),
        )
        self.aff1 = layers.AFF(base_channels)
        self.aff_conv2 = nn.Sequential(
            Conv2d_BN(base_channels * 2, base_channels,
                      3, stride=1, padding=1, bias=False),
            nn.ReLU(),
        )
        self.aff_conv1 = nn.Sequential(
            Conv2d_BN(base_channels, base_channels, 3,
                      stride=1, padding=1, bias=False),
            nn.ReLU(),
        )

        self.side_l3 = Conv2d_BN(base_channels * 2, num_classes, 1)
        self.side_l2 = Conv2d_BN(base_channels, num_classes, 1)
        self.side_l1 = Conv2d_BN(base_channels, num_classes, 1)

        self.alpha_l3 = alphas[2]
        self.alpha_l2 = alphas[1]
        self.alpha_l1 = alphas[0]
        self.upscale_rate = upscale_rate
        self.super_resolution = super_resolution
        self.output_size = output_size

        if super_resolution:
            self.sr_aff3 = layers.AFF(base_channels * 4)
            self.sr_aff3_conv = nn.Sequential(
                Conv2d_BN(base_channels * 4, base_channels *
                          2, 3, 1, padding=1, bias=False),
                nn.ReLU(),
            )

            self.sr_aff2 = layers.AFF(base_channels * 2)
            self.sr_aff2_conv = nn.Sequential(
                Conv2d_BN(base_channels * 2, base_channels,
                          3, 1, padding=1, bias=False),
                nn.ReLU(),
            )

            self.sr_aff1 = layers.AFF(base_channels)
            self.sr_aff1_conv = nn.Sequential(
                Conv2d_BN(base_channels, base_channels,
                          3, 1, padding=1, bias=False),
                nn.ReLU(),
            )

            self.sr = nn.Sequential(
                Conv2d_BN(base_channels, 64, kernel_size=5,
                          stride=1, padding=2, bias=False),
                nn.Tanh(),
                Conv2d_BN(64, 32, kernel_size=3, stride=1,
                          padding=1, bias=False),
                nn.Tanh(),
                Conv2d_BN(32, (upscale_rate ** 2) * in_channels,
                          kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(upscale_factor=upscale_rate),
            )
            self.query = layers.FIM(in_channels, num_classes, hidden_state=16)

    def forward(self, x):
        en1 = self.encoder1(x)
        down1 = self.down(en1)
        en2 = self.encoder2(down1)
        down2 = self.down(en2)
        en3 = self.encoder3(down2)
        down3 = self.down(en3)

        sfa = self.sfa(down3)
        sfa = functional.interpolate(
            sfa, size=en3.shape[2:], mode="bilinear", align_corners=True)
        aff3 = self.aff3(en3, sfa)
        aff3 = self.aff_conv3(aff3)
        aff3_up = functional.interpolate(
            aff3, size=en2.shape[2:], mode="bilinear", align_corners=True)
        aff2 = self.aff2(en2, aff3_up)
        aff2 = self.aff_conv2(aff2)
        aff2_up = functional.interpolate(
            aff2, size=en1.shape[2:], mode="bilinear", align_corners=True)
        aff1 = self.aff1(en1, aff2_up)
        aff1 = self.aff_conv1(aff1)

        side1 = self.side_l1(aff1)
        side2 = self.side_l2(functional.interpolate(
            aff2, size=x.shape[2:], mode="bilinear", align_corners=True))
        side3 = self.side_l3(functional.interpolate(
            aff3, size=x.shape[2:], mode="bilinear", align_corners=True))

        out = self.alpha_l1*side1 + self.alpha_l2*side2 + self.alpha_l3*side3
        if self.super_resolution and (self.output_size is not None):
            out = functional.interpolate(
                out, size=self.output_size, mode="bilinear", align_corners=True)

        sr = None
        qr_seg = None
        if self.super_resolution:
            out = functional.interpolate(
                out, scale_factor=self.upscale_rate, mode="bilinear", align_corners=True)

            if self.training:
                sr_aff3 = self.sr_aff3(en3, sfa)
                sr_aff3 = self.sr_aff3_conv(sr_aff3)
                sr_aff3_up = functional.interpolate(
                    sr_aff3, size=en2.shape[2:], mode="bilinear", align_corners=True)
                sr_aff2 = self.sr_aff2(en2, sr_aff3_up)
                sr_aff2 = self.sr_aff2_conv(sr_aff2)
                sr_aff2_up = functional.interpolate(
                    sr_aff2, size=en1.shape[2:], mode="bilinear", align_corners=True)
                sr_aff1 = self.sr_aff1(en1, sr_aff2_up)
                aff1 = self.sr_aff1_conv(sr_aff1)
                sr = self.sr(aff1)

                if self.output_size is not None:
                    sr = functional.interpolate(
                        sr, size=self.output_size, mode="bilinear", align_corners=True)

        if self.super_resolution and self.training:
            qr_seg = self.query(sr, out)
        if sr is None:
            return out
        return out, qr_seg


def check():
    model = SCSNet(in_channels=3, num_classes=1)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    out = model(torch.rand(2, 3, 128, 128))
    print(out.shape)

    model = SCSNet(in_channels=3, num_classes=1,
                   super_resolution=True, upscale_rate=1)
    model.train()
    outs = model(torch.rand(2, 3, 128, 128))
    print([(out.shape, out.min(), out.max()) for out in outs])


if __name__ == "__main__":
    check()
