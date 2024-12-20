import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNPReLU(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        args:
        nIn: number of input channels
        nOut: number of output channels
        kSize: kernel size
        stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv2d(
            nIn, nOut, (kSize, kSize), stride=stride,
            padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        """
        args:
        input: input feature map
        return: transformed feature map
        """
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output


class BNPReLU(nn.Module):
    def __init__(self, nOut):
        """
        args:
        nOut: channels of output feature maps
        """
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        """
        args:
        input: input feature map
        return: normalized and thresholded feature map
        """
        output = self.bn(input)
        output = self.act(output)
        return output


class ConvBN(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        args:
        nIn: number of input channels
        nOut: number of output channels
        kSize: kernel size
        stride: optinal stide for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv2d(
            nIn, nOut, (kSize, kSize), stride=stride,
            padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)

    def forward(self, input):
        """
        args:
        input: input feature map
        return: transformed feature map
        """
        output = self.conv(input)
        output = self.bn(output)
        return output


class ChannelWiseConv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        Args:
        nIn: number of input channels
        nOut: number of output channels, default (nIn == nOut)
        kSize: kernel size
        stride: optional stride rate for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv2d(
            nIn, nOut, (kSize, kSize), stride=stride,
            padding=(padding, padding), groups=nIn, bias=False)

    def forward(self, input):
        """
        args:
        input: input feature map
        return: transformed feature map
        """
        output = self.conv(input)
        return output


class DilatedConv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        """
        args:
        nIn: number of input channels
        nOut: number of output channels
        kSize: kernel size
        stride: optional stride rate for down-sampling
        d: dilation rate
        """
        super().__init__()
        padding = int((kSize - 1)/2) * d
        self.conv = nn.Conv2d(
            nIn, nOut, (kSize, kSize), stride=stride,
            padding=(padding, padding), bias=False, dilation=d)

    def forward(self, input):
        """
        args:
        input: input feature map
        return: transformed feature map
        """
        output = self.conv(input)
        return output


class ChannelWiseDilatedConv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        """
        args:
        nIn: number of input channels
        nOut: number of output channels, default (nIn == nOut)
        kSize: kernel size
        stride: optional stride rate for down-sampling
        d: dilation rate
        """
        super().__init__()
        padding = int((kSize - 1)/2) * d
        self.conv = nn.Conv2d(
            nIn, nOut, (kSize, kSize), stride=stride,
            padding=(padding, padding), groups=nIn, bias=False, dilation=d)

    def forward(self, input):
        """
        args:
        input: input feature map
        return: transformed feature map
        """
        output = self.conv(input)
        return output


class FGlo(nn.Module):
    """
    the FGlo class is employed to refine the joint feature of both local feature and surrounding context.
    """

    def __init__(self, channel, reduction=16):
        super(FGlo, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
# ContextGuidedBlock


class CG_Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CG_Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.CG_moudle = CG_moudle(out_ch, out_ch, add=True)
        self.ca = ChannelAttention(out_ch)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.conv(x)
        x = self.CG_moudle(x)
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


class CG_moudle1(nn.Module):
    def __init__(self, nIn, nOut, dilation_rate=2, reduction=16, add=False):
        """
        args:
        nIn: number of input channels
        nOut: number of output channels,
        add: if true, residual learning
        """
        super().__init__()
        n = int(nOut / 2)
        # 1x1 Conv is employed to reduce the computation
        self.conv1x1 = ConvBNPReLU(nIn, n, 1, 1)
        self.F_loc = ChannelWiseConv(n, n, 3, 1)  # local feature
        self.F_sur = ChannelWiseDilatedConv(
            n, n, 3, 1, dilation_rate)  # surrounding context
        self.bn_prelu = BNPReLU(nOut)
        self.add = add
        self.F_glo = FGlo(nOut, reduction)

    def forward(self, input):
        output = self.conv1x1(input)
        loc = self.F_loc(output)
        sur = self.F_sur(output)

        joi_feat = torch.cat([loc, sur], 1)

        joi_feat = self.bn_prelu(joi_feat)

        # F_glo is employed to refine the joint feature
        output = self.F_glo(joi_feat)
        # if residual version
        if self.add:
            output = input + output
        return output


class CG_moudle(nn.Module):
    def __init__(self, nIn, nOut, dilation_rate=2, reduction=16, add=False):
        """
        args:
        nIn: number of input channels
        nOut: number of output channels,
        add: if true, residual learning
        """
        super().__init__()
        n = int(nOut / 2)
        # 1x1 Conv is employed to reduce the computation
        self.conv1x1 = ConvBNPReLU(nIn, n, 1, 1)
        self.F_loc = ChannelWiseConv(n, n, 3, 1)  # local feature
        self.F_sur = ChannelWiseDilatedConv(
            n, n, 3, 1, dilation_rate)  # surrounding context
        self.bn_prelu = BNPReLU(nOut)
        self.add = add

    def forward(self, input):
        output = self.conv1x1(input)
        loc = self.F_loc(output)
        sur = self.F_sur(output)
        joi_feat = torch.cat([loc, sur], 1)
        output = self.bn_prelu(joi_feat)
        # if residual version
        if self.add:
            output = input + output
        return output


class CGBM_block2(nn.Module):
    def __init__(self, in_ch, out_ch, dilation_rate=2):
        super(CGBM_block2, self).__init__()
        n = int(out_ch / 2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        # 1x1 Conv is employed to reduce the computation
        self.conv1x1 = ConvBNPReLU(out_ch, n, 1, 1)
        self.F_loc = ChannelWiseConv(n, n, 3, 1)  # local feature
        self.F_sur = ChannelWiseDilatedConv(
            n, n, 3, 1, dilation_rate)  # surrounding context
        self.bn_prelu = BNPReLU(out_ch)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.conv_fushion = ConvBNPReLU(out_ch*2, out_ch, 1, 1)
        self.ca = ChannelAttention(out_ch)
        self.sa_or = SpatialAttention()
        self.sa_loc = SpatialAttention()
        self.sa_sur = SpatialAttention()

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.conv2(x)
        x = self.conv1x1(x1)
        loc = self.F_loc(x)
        sur = self.F_sur(x)
        joi_feat = torch.cat([loc, sur, x1], 1)
        x = self.conv_fushion(joi_feat)
        x = x * self.ca(x)
        x = x * (self.sa_loc(loc)+self.sa_sur(sur)+self.sa_sur(x))
        return x


class CGBM_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CGBM_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.CG_moudle = CG_moudle(out_ch, out_ch, add=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.CG_moudle(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
# _nomal


class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class CBAM_Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CBAM_Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.ca = ChannelAttention(out_ch)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.conv(x)
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, block=double_conv):
        super(inconv, self).__init__()
        self.conv = block(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, block=double_conv):
        super(down, self).__init__()
        self.max_pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            block(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.max_pool_conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True, block=double_conv):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = block(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(
            x2,
            (diffX // 2, int(diffX / 2), diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class CBAM_UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(CBAM_UNet, self).__init__()
        self.inc = inconv(n_channels, 64, block=double_conv)
        self.down1 = down(64, 128, block=CBAM_Block)
        self.down2 = down(128, 256, block=CBAM_Block)
        self.down3 = down(256, 512, block=CBAM_Block)
        self.down4 = down(512, 512, block=CBAM_Block)
        self.up1 = up(1024, 256, block=CBAM_Block)
        self.up2 = up(512, 128, block=CBAM_Block)
        self.up3 = up(256, 64, block=CBAM_Block)
        self.up4 = up(128, 64, block=CBAM_Block)
        self.outc = outconv(64, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64, block=double_conv)
        self.down1 = down(64, 128, block=double_conv)
        self.down2 = down(128, 256, block=double_conv)
        self.down3 = down(256, 512, block=double_conv)
        self.down4 = down(512, 512, block=double_conv)
        self.up1 = up(1024, 256, block=double_conv)
        self.up2 = up(512, 128, block=double_conv)
        self.up3 = up(256, 64, block=double_conv)
        self.up4 = up(128, 64, block=double_conv)
        self.outc = outconv(64, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        # x = self.relu(x)
        return x


class CG_UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(CG_UNet, self).__init__()
        self.inc = inconv(n_channels, 64, block=CG_Block)
        self.down1 = down(64, 128, block=CG_Block)
        self.down2 = down(128, 256, block=CG_Block)
        self.down3 = down(256, 512, block=CG_Block)
        self.down4 = down(512, 512, block=CG_Block)
        self.up1 = up(1024, 256, block=CG_Block)
        self.up2 = up(512, 128, block=CG_Block)
        self.up3 = up(256, 64, block=CG_Block)
        self.up4 = up(128, 64, block=CG_Block)
        self.outc = outconv(64, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


class CGAM_UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(CGAM_UNet, self).__init__()
        self.inc = inconv(n_channels, 64, block=CGBM_block)
        self.down1 = down(64, 128, block=CGBM_block)
        self.down2 = down(128, 256, block=CGBM_block)
        self.down3 = down(256, 512, block=CGBM_block)
        self.down4 = down(512, 512, block=CGBM_block)
        self.up1 = up(1024, 256, block=CGBM_block)
        self.up2 = up(512, 128, block=CGBM_block)
        self.up3 = up(256, 64, block=CGBM_block)
        self.up4 = up(128, 64, block=CGBM_block)
        self.outc = outconv(64, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


class CGAM_UNet2(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(CGAM_UNet2, self).__init__()
        self.inc = inconv(n_channels, 64, block=double_conv)
        self.down1 = down(64, 128, block=CGBM_block2)
        self.down2 = down(128, 256, block=CGBM_block2)
        self.down3 = down(256, 512, block=CGBM_block2)
        self.down4 = down(512, 512, block=CGBM_block2)
        self.up1 = up(1024, 256, block=CGBM_block2)
        self.up2 = up(512, 128, block=CGBM_block2)
        self.up3 = up(256, 64, block=CGBM_block2)
        self.up4 = up(128, 64, block=CGBM_block2)
        self.outc = outconv(64, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


def check():
    model = CGAM_UNet2(n_channels=3, n_classes=1)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    out = model(torch.rand(2, 3, 128, 128))
    print(out.shape, out.min(), out.max())


if __name__ == "__main__":
    check()
