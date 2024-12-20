import math
import torch
import torch.nn as nn
import torch.nn.functional as functional
from einops import rearrange

from mlpipeline.models.segmentation.danet.vit import pair


def get_strip_backward_hook(eye):
    def hook(grad):
        grad = grad * rearrange(eye, "h w -> 1 1 h w")
        return grad
    return hook


class StripConv2d(nn.Module):
    def __init__(self, strip_type, in_channels, out_channels, kernel_size, stride):
        """Define a strip convolution

        Args:
            strip_type (str): type of strip, either 'upward' (45 degree) or 'downward' (-45 degree)
            in_channels (int):
            out_channels (int):
            kernel_size (int):
            stride (int):
        """
        super(StripConv2d, self).__init__()
        assert (strip_type in ["upward", "downward"])

        self.strip_type = strip_type
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = 1

        self.eye = nn.Parameter(torch.eye(kernel_size[0], requires_grad=False))
        if strip_type == "upward":
            self.eye = nn.Parameter(torch.rot90(self.eye))

        kernel_size = pair(kernel_size)
        self.weight = nn.Parameter(torch.zeros(
            out_channels, in_channels, *kernel_size))

        self.reset_parameters()
        self.weight.register_hook(get_strip_backward_hook(self.eye))

    def reset_parameters(self):
        with torch.no_grad():
            weight = torch.empty_like(self.weight)
            nn.init.kaiming_normal_(weight, a=math.sqrt(5))
            weight = weight * rearrange(self.eye, "h w -> 1 1 h w")
            self.weight.copy_(weight)
        return

    def forward(self, inputs):
        out = functional.conv2d(
            functional.pad(
                inputs,
                tuple([self.padding] * 4),
                mode="constant"),
            self.weight,
            None,
            self.stride,
            pair(0),
            1,
            1)
        return out


class Conv2d_BN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=(1, 1),
        dilation=1,
        bias=True,
        use_bn=True,
        strip_type=None,
    ):
        super(Conv2d_BN, self).__init__()
        assert (strip_type in ["upward", "downward"]) or (strip_type is None)
        kernel_size = pair(kernel_size)

        padding = (
            (dilation * (kernel_size[0] - 1)) // 2, (dilation * (kernel_size[1] - 1)) // 2)
        if strip_type is None:
            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias,
            )
        else:
            self.conv = StripConv2d(
                strip_type=strip_type,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
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


class VggBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
    ):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = Conv2d_BN(in_channels, out_channels, kernel_size)
        self.conv2 = Conv2d_BN(out_channels, out_channels, kernel_size)

    def forward(self, inputs):
        x = self.relu(self.conv1(inputs))
        x = self.relu(self.conv2(x))
        return x


class VGG(nn.Module):
    def __init__(self, in_channels, features_dim_list):
        super(VGG, self).__init__()
        self.max_pool = nn.MaxPool2d((3, 3), stride=2, padding=1)
        self.block1 = VggBlock(in_channels, features_dim_list[0])
        self.block2 = VggBlock(features_dim_list[0], features_dim_list[1])
        self.block3 = VggBlock(features_dim_list[1], features_dim_list[2])
        self.block4 = VggBlock(features_dim_list[2], features_dim_list[3])
        self.block5 = VggBlock(features_dim_list[3], features_dim_list[4])

    def forward(self, inputs):
        x1 = self.block1(inputs)
        x2 = self.block2(self.max_pool(x1))
        x3 = self.block3(self.max_pool(x2))
        x4 = self.block4(self.max_pool(x3))
        x5 = self.block5(self.max_pool(x4))
        return x1, x2, x3, x4, x5


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes, planes,
        stride=1, downsample=None, groups=1,
        base_width=64, dilation=1, norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                "BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = Conv2d_BN(in_planes, planes, 3, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv2d_BN(planes, planes, 3)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self, in_planes, planes,
        stride=1, downsample=None, groups=1,
        base_width=64, dilation=1, norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = Conv2d_BN(in_planes, width, 1)
        self.conv2 = Conv2d_BN(
            width, width, 3, stride=stride, dilation=dilation)
        self.conv3 = Conv2d_BN(width, planes * self.expansion, 1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self, block, layers, groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.in_planes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        assert len(replace_stride_with_dilation) == 3

        self.groups = groups
        self.base_width = width_per_group

        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv1 = Conv2d_BN(3, self.in_planes, 7, stride=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2,
            dilate=replace_stride_with_dilation[0])

        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2,
            dilate=replace_stride_with_dilation[1])

        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2,
            dilate=replace_stride_with_dilation[2])

        self.conv1_out = Conv2d_BN(64, 32, 1)
        self.conv2_out = Conv2d_BN(256, 64, 1)
        self.conv3_out = Conv2d_BN(512, 96, 1)
        self.conv4_out = Conv2d_BN(1024, 128, 1)
        self.conv5_out = Conv2d_BN(2048, 196, 1)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = Conv2d_BN(
                self.in_planes, planes * block.expansion, 1, stride=stride)

        layers = [
            block(
                self.in_planes, planes, stride, downsample, self.groups,
                self.base_width, previous_dilation, norm_layer),
        ]
        self.in_planes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(
                    self.in_planes, planes, groups=self.groups,
                    base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        y1 = out = self.relu(out)
        out = self.max_pool(out)

        y2 = out = self.layer1(out)
        y3 = out = self.layer2(out)

        y4 = out = self.layer3(out)
        y5 = out = self.layer4(out)

        x1 = self.conv1_out(y1)
        x2 = self.conv2_out(y2)
        x3 = self.conv3_out(y3)
        x4 = self.conv4_out(y4)
        x5 = self.conv5_out(y5)
        return x1, x2, x3, x4, x5


def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def resnet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])
