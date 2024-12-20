import torch
import torch.nn as nn
import torch.nn.functional as functional


def swish(x):
    return x * functional.relu6(x + 3) / 6


class BasicConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels,
        kernel_size=3,
        stride=1, padding=1,
        dilation=1, groups=1,
        bias=False, bn=True,
        activation=swish, conv=nn.Conv2d,
    ):
        super(BasicConv2d, self).__init__()
        if not isinstance(kernel_size, tuple):
            if dilation > 1:
                # AtrousConv2d
                padding = dilation * (kernel_size // 2)
            elif kernel_size == stride:
                padding = 0
            else:
                # BasicConv2d
                padding = kernel_size // 2

        self.c = conv(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride, padding=padding,
            dilation=dilation, bias=bias,
        )
        self.b = nn.BatchNorm2d(
            out_channels, momentum=0.01) if bn else nn.Identity()

        drop_prob = 0.15
        self.o = nn.Dropout2d(p=drop_prob, inplace=False)

        if activation is None:
            self.a = nn.Identity()
        else:
            self.a = activation
        return

    def forward(self, x):
        x = self.c(x)
        x = self.o(x)
        x = self.b(x)
        x = self.a(x)
        return x


class Bottleneck(nn.Module):
    MyConv = BasicConv2d

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **args):
        super(Bottleneck, self).__init__()

        self.conv1 = self.MyConv(
            in_channels, out_channels, kernel_size=3, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        # nn.LeakyReLU()
        self.relu = swish
        if (downsample is None) and (in_channels != out_channels):
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        self.downsample = downsample

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.relu(out + residual)
        return out


class ConvBlock(torch.nn.Module):
    attention = None
    MyConv = BasicConv2d

    def __init__(self, in_channels, out_channels, kernel_size=3, shortcut=False, pool=True):
        super(ConvBlock, self).__init__()
        self.shortcut = nn.Sequential(nn.Conv2d(
            in_channels, out_channels, kernel_size=1), nn.BatchNorm2d(out_channels))
        padding = (kernel_size - 1) // 2

        if pool:
            self.pool = nn.MaxPool2d(kernel_size=2)
        else:
            self.pool = None

        block = []
        block.append(self.MyConv(in_channels, out_channels,
                     kernel_size=kernel_size, padding=padding))

        block.append(self.MyConv(out_channels, out_channels,
                     kernel_size=kernel_size, padding=padding))
        self.block = nn.Sequential(*block)

    def forward(self, x):
        if self.pool is not None:
            x = self.pool(x)

        out = self.block(x)
        return swish(out + self.shortcut(x))


class OutSigmoid(nn.Module):
    def __init__(self, in_channels, channels=8):
        super(OutSigmoid, self).__init__()
        self.cls = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Conv2d(in_channels=channels, out_channels=1,
                      kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.cls(x)
