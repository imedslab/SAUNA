from typing import Callable

import torch
import torch.nn as nn


class Conv2d_BN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=(1, 1),
        dilation=1,
        use_bn=True,
    ):
        super(Conv2d_BN, self).__init__()
        padding = (dilation * (kernel_size - 1)) // 2
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        if use_bn:
            self.batchnorm = nn.BatchNorm2d(out_channels)
        else:
            self.batchnorm = None
        return

    def forward(self, inputs):
        x = self.conv(inputs)
        if self.batchnorm is not None:
            x = self.batchnorm(x)
        return x


class SqueezeExcitation(nn.Module):

    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        activation: Callable[..., nn.Module] = nn.ReLU(),
        scale_activation: Callable[..., nn.Module] = nn.Sigmoid(),
    ) -> None:
        super(SqueezeExcitation, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = activation
        self.scale_activation = scale_activation

    def _scale(self, input: torch.Tensor) -> torch.Tensor:
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        scale = self._scale(input)
        return scale * input


class SpatialAttention(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.softmax = nn.Softmax2d()

    def forward(self, inputs):
        x = self.conv(inputs)
        weights = self.softmax(x)

        return (weights * inputs)
