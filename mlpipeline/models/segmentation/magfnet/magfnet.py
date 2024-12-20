import torch
import torch.nn as nn
import torch.nn.functional as functional

import mlpipeline.models.segmentation.magfnet.base as base
from mlpipeline.models.segmentation.magfnet.encoder_decoder import Encoder, Decoder


class RB(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(RB, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv11 = base.Conv2d_BN(in_channels, in_channels, kernel_size)
        self.conv12 = base.Conv2d_BN(in_channels, in_channels, kernel_size)
        self.conv21 = base.Conv2d_BN(in_channels, in_channels, 1)

    def forward(self, inputs):
        x1 = self.relu(self.conv11(inputs))
        x1 = self.relu(self.conv12(x1))
        x2 = self.relu(self.conv21(inputs))
        out = x1 + x2
        return out


class FE(nn.Module):
    def __init__(self, in_channels):
        super(FE, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.branch1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            base.Conv2d_BN(in_channels, in_channels, kernel_size=1),
            self.relu,
            nn.ConvTranspose2d(in_channels, in_channels,
                               kernel_size=2, stride=2),
        )
        self.branch2 = base.Conv2d_BN(
            in_channels, in_channels, kernel_size=3, dilation=2)
        self.branch3 = base.Conv2d_BN(
            in_channels, in_channels, kernel_size=3, dilation=3)
        self.branch4 = base.Conv2d_BN(
            in_channels, in_channels, kernel_size=3, dilation=4)

        self.conv = base.Conv2d_BN(in_channels * 4, in_channels, kernel_size=1)

    def forward(self, inputs):
        x1 = self.branch1(inputs)
        x2 = self.relu(self.branch2(inputs))
        x3 = self.relu(self.branch3(inputs))
        x4 = self.relu(self.branch4(inputs))

        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.conv(x)
        return x


class AGF(nn.Module):
    def __init__(self, in_channels_main, list_in_channels_sp):
        super(AGF, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = base.Conv2d_BN(
            in_channels_main, in_channels_main, kernel_size=3)
        self.se1 = base.SqueezeExcitation(
            in_channels_main, in_channels_main // 2, self.relu)

        self.convs = nn.ModuleList()
        for in_channels_sp in list_in_channels_sp:
            out_dim = in_channels_main if len(
                list_in_channels_sp) == 1 else (in_channels_main // 2)
            conv_i = base.Conv2d_BN(in_channels_sp, out_dim, kernel_size=3)
            self.convs.append(conv_i)

        self.conv2 = nn.Sequential(
            base.Conv2d_BN(int(in_channels_main * 1.5),
                           in_channels_main, kernel_size=1),
            self.relu,
        ) if len(list_in_channels_sp) == 3 else nn.Identity()
        self.attn2 = base.SpatialAttention(in_channels_main, kernel_size=3)

    def forward(self, main_inputs, list_inputs):
        x = self.relu(self.conv1(main_inputs))
        x = self.se1(x)

        list_x_i = []
        for index, x_i in enumerate(list_inputs):
            x_i = functional.interpolate(x_i, size=(x.shape[2], x.shape[3]))
            x_i = self.relu(self.convs[index](x_i))
            list_x_i.append(x_i)

        x_i = torch.cat(list_x_i, dim=1)
        x_i = self.conv2(x_i)
        x_i = self.attn2(x_i)

        out = x + x_i
        return out


class MagfNet(nn.Module):
    def __init__(self, in_channels, num_classes, features_dim=64):
        super(MagfNet, self).__init__()

        self.encoder = Encoder(in_channels=in_channels,
                               features_dim=features_dim, stride=2)
        self.decoder = Decoder(features_dim=features_dim,
                               out_channels=num_classes, stride=2)

        self.rb1 = RB(features_dim, kernel_size=3)
        self.rb2 = RB(features_dim * 2, kernel_size=3)
        self.rb3 = RB(features_dim * 4, kernel_size=3)
        self.rb4 = RB(features_dim * 8, kernel_size=3)
        self.fe = FE(features_dim * 8)

        self.agf2 = AGF(features_dim * 2, [features_dim])
        self.agf3 = AGF(features_dim * 4, [features_dim, features_dim * 2])
        self.agf4 = AGF(features_dim * 8,
                        [features_dim, features_dim * 2, features_dim * 4])

    def forward(self, inputs):
        x1, x2, x3, x4 = self.encoder(inputs)

        x1 = self.rb1(x1)
        x2 = self.rb2(x2)
        x3 = self.rb3(x3)
        x4 = self.fe(self.rb4(x4))

        x2f = self.agf2(x2, [x1])
        x3f = self.agf3(x3, [x1, x2])
        x4f = self.agf4(x4, [x1, x2, x3])

        out = self.decoder(x4, x4f, x3f, x2f)
        return out


def check():
    model = MagfNet(in_channels=3, num_classes=1, features_dim=64)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    out = model(torch.rand(2, 3, 128, 128))
    print(out.shape, out.min(), out.max())


if __name__ == "__main__":
    check()
