import torch
import torch.nn as nn

import mlpipeline.models.segmentation.skelcon.model_basic as model_basic


class UpsampleBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, up_mode="transp_conv"):
        super(UpsampleBlock, self).__init__()
        block = []
        if up_mode == "transp_conv":
            block.append(nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2))
        elif up_mode == "up_conv":
            block.append(nn.Upsample(mode="bilinear",
                         scale_factor=2, align_corners=False))
            block.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        else:
            raise Exception("Upsampling mode not supported")

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class ConvBridgeBlock(torch.nn.Module):
    def __init__(self, out_channels, kernel_size=3):
        super(ConvBridgeBlock, self).__init__()
        padding = (kernel_size - 1) // 2
        block = []

        block.append(nn.Conv2d(out_channels, out_channels,
                     kernel_size=kernel_size, padding=padding))
        block.append(nn.LeakyReLU())
        block.append(nn.BatchNorm2d(out_channels))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UpConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, up_mode="up_conv", conv_bridge=False, shortcut=False):
        super(UpConvBlock, self).__init__()

        self.conv_bridge = conv_bridge
        if self.conv_bridge:
            self.conv_bridge_layer = ConvBridgeBlock(
                out_channels, kernel_size=kernel_size)

        self.up_layer = UpsampleBlock(
            in_channels, out_channels, up_mode=up_mode)
        self.conv_layer = model_basic.ConvBlock(
            2 * out_channels,
            out_channels,
            kernel_size=kernel_size,
            shortcut=shortcut,
            pool=False,
        )

    def forward(self, x, skip):
        up = self.up_layer(x)
        if self.conv_bridge:
            out = torch.cat([up, self.conv_bridge_layer(skip)], dim=1)
        else:
            out = torch.cat([up, skip], dim=1)

        out = self.conv_layer(out)
        return out


class LUNet(nn.Module):
    __name__ = "lunet"
    use_render = False

    def __init__(self, in_channels=1, n_classes=1, layers=(32, 32, 32, 32, 32), num_emb=128):
        super(LUNet, self).__init__()
        self.num_features = layers[-1]

        self.__name__ = "u{}x{}".format(len(layers), layers[0])
        self.n_classes = n_classes
        self.first = model_basic.BasicConv2d(in_channels, layers[0])

        self.down_path = nn.ModuleList()
        for i in range(len(layers) - 1):
            block = model_basic.ConvBlock(
                in_channels=layers[i], out_channels=layers[i + 1], pool=True)
            self.down_path.append(block)

        self.up_path = nn.ModuleList()
        reversed_layers = list(reversed(layers))
        for i in range(len(layers) - 1):
            block = UpConvBlock(
                in_channels=reversed_layers[i], out_channels=reversed_layers[i + 1])
            self.up_path.append(block)

        self.conv_bn = nn.Sequential(
            nn.Conv2d(layers[0], layers[0], kernel_size=1),
            nn.BatchNorm2d(layers[0]),
        )
        self.aux = nn.Sequential(
            nn.Conv2d(layers[0], n_classes, kernel_size=1),
            # nn.BatchNorm2d(n_classes),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.first(x)

        down_activations = []
        for i, down in enumerate(self.down_path):
            down_activations.append(x)
            x = down(x)

        down_activations.reverse()

        for i, up in enumerate(self.up_path):
            x = up(x, down_activations[i])

        # self.feat = functional.normalize(x, dim=1, p=2)
        x = self.conv_bn(x)
        self.feat = x

        self.pred = self.aux(x)
        return self.pred


def lunet(**args):
    model_basic.ConvBlock.attention = None
    net = LUNet(**args)
    net.__name__ = "lunet"
    return net
