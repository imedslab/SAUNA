import torch
import torch.nn as nn
import torch.nn.functional as functional


class Conv2d_BN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        use_bn=True,
    ):
        super(Conv2d_BN, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
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


class SFA(nn.Module):
    def __init__(self, in_channels):
        """
            Implementation of Scale-aware feature aggregation module
            References:
                "SCS-Net: A Scale and Context Sensitive Network for Retinal Vessel Segmentation"

            Parameters:
                in_channels (int): number of input channels
        """
        super(SFA, self).__init__()
        self.conv3x3_1 = Conv2d_BN(
            in_channels, in_channels, 3, stride=1, padding=1, dilation=1, bias=False)
        self.conv3x3_2 = Conv2d_BN(
            in_channels, in_channels, 3, stride=1, padding=3, dilation=3, bias=False)
        self.conv3x3_3 = Conv2d_BN(
            in_channels, in_channels, 3, stride=1, padding=5, dilation=5, bias=False)

        self.conv3x3_12 = nn.Sequential(
            Conv2d_BN(in_channels * 2, in_channels, 3,
                      stride=1, padding=1, bias=False),
            nn.ReLU(),
        )

        self.conv3x3_23 = nn.Sequential(
            Conv2d_BN(in_channels * 2, in_channels, 3,
                      stride=1, padding=1, bias=False),
            nn.ReLU(),
        )

        self.conv1x1_12 = Conv2d_BN(in_channels, 2, 1, bias=False)
        self.conv1x1_23 = Conv2d_BN(in_channels, 2, 1, bias=False)

        self.out_conv = nn.Sequential(
            Conv2d_BN(in_channels, in_channels, 1, bias=False),
            nn.ReLU(),
        )

    def forward(self, x):
        d1 = self.conv3x3_1(x)
        d2 = self.conv3x3_2(x)
        d3 = self.conv3x3_3(x)

        f12 = torch.cat([d1, d2], dim=1)
        f12 = self.conv3x3_12(f12)
        f12 = self.conv1x1_12(f12)

        f12 = torch.softmax(f12, dim=1)
        w_as = torch.chunk(f12, 2, dim=1)

        w_f1 = w_as[0] * d1
        w_f2 = w_as[1] * d2
        w_f12 = w_f1 + w_f2

        f23 = torch.cat([d2, d3], dim=1)
        f23 = self.conv3x3_23(f23)
        f23 = self.conv1x1_23(f23)
        f23 = torch.softmax(f23, dim=1)
        w_bs = torch.chunk(f23, 2, dim=1)

        w_f2_1 = w_bs[0] * d2
        w_f3 = w_bs[1] * d3

        w_f23 = w_f2_1 + w_f3
        out = w_f12 + w_f23 + x
        out = self.out_conv(out)
        return out


class AFF(nn.Module):
    def __init__(self, in_channels, reduction=16):
        """
            Implementation of Adaptive feature fusion module
            References:
                "SCS-Net: A Scale and Context Sensitive Network for Retinal Vessel Segmentation"
            Parameters
            ----------
            in_channels (int): number of channels of input
            reduction (int): reduction rate for squeeze
        """
        super(AFF, self).__init__()
        in_ch = in_channels * 2
        hidden_ch = (in_channels * 2) // reduction
        self.se = nn.Sequential(
            Conv2d_BN(in_ch, hidden_ch, 1),
            nn.ReLU(),
            Conv2d_BN(hidden_ch, in_ch, 1),
            nn.Sigmoid(),
        )
        self.conv1x1 = Conv2d_BN(in_ch, in_channels, 1)

    def forward(self, x1, x2):
        """

        Parameters
        ----------
        x1 (Tensor): low level feature, (n,c,h,w)
        x2 (Tensor): high level feature, (n,c,h,w)

        Returns
        -------
            Tensor, fused feature
        """
        x12 = torch.cat([x1, x2], dim=1)
        se = self.se(x12)
        se = self.conv1x1(se)
        se = functional.adaptive_avg_pool2d(se, 1)
        se = torch.sigmoid(se)
        w1 = se * x1
        out = w1 + x2
        return out


class SpatialModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpatialModule, self).__init__()
        hidden_state = out_channels * 3
        self.conv1 = nn.Sequential(
            Conv2d_BN(in_channels, hidden_state, 1, 1),
            nn.ReLU(),
        )
        self.branch1 = Conv2d_BN(
            out_channels, out_channels, 3, 1, padding=1, dilation=1, groups=out_channels)
        self.branch2 = Conv2d_BN(
            out_channels, out_channels, 3, 1, padding=2, dilation=2, groups=out_channels)
        self.branch3 = Conv2d_BN(
            out_channels, out_channels, 3, 1, padding=4, dilation=4, groups=out_channels)
        self.shuffle = nn.ChannelShuffle(3)
        self.fusion = nn.Sequential(
            Conv2d_BN(hidden_state, out_channels, 1, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        net = self.conv1(x)
        splits = torch.chunk(net, 3, dim=1)
        branch1 = self.branch1(splits[0])
        branch2 = self.branch2(splits[1])
        branch3 = self.branch3(splits[2])
        net = torch.cat([branch1, branch2, branch3], dim=1)

        net = self.shuffle(net)
        net = self.fusion(net)
        return net


class FIM(nn.Module):
    def __init__(self, in_ch1, in_ch2, hidden_state=16):
        super(FIM, self).__init__()
        self.conv1 = SpatialModule(in_ch1 + in_ch2, hidden_state)
        self.conv2 = nn.Sequential(
            Conv2d_BN(hidden_state, 1, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, f1, f2):
        concat = torch.cat([f1, f2], dim=1)
        net = self.conv1(concat)
        net = self.conv2(net)
        out_seg = net * f2 + f2
        return out_seg
