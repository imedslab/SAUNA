import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, dp=0):
        super(Conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dp),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dp),
            nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        return self.conv(x)


class FeatureFuse(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureFuse, self).__init__()
        self.conv11 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, padding=0, bias=False)
        self.conv33 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, padding=1, bias=False)
        self.conv33_di = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, padding=2, bias=False, dilation=2)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x1 = self.conv11(x)
        x2 = self.conv33(x)
        x3 = self.conv33_di(x)
        out = self.norm(x1 + x2 + x3)
        return out


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, dp=0):
        super(Up, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2,
                padding=0, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=False))

    def forward(self, x):
        x = self.up(x)
        return x


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, dp=0):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=2,
                padding=0, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        x = self.down(x)
        return x


class Block(nn.Module):
    def __init__(
        self, in_channels, out_channels,
        dp=0, is_up=False, is_down=False,
        fuse=False,
    ):
        super(Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if fuse:
            self.fuse = FeatureFuse(in_channels, out_channels)
        else:
            self.fuse = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1)

        self.is_up = is_up
        self.is_down = is_down
        self.conv = Conv(out_channels, out_channels, dp=dp)

        if self.is_up:
            self.up = Up(out_channels, out_channels // 2)
        if self.is_down:
            self.down = Down(out_channels, out_channels * 2)

    def forward(self,  x):
        if self.in_channels != self.out_channels:
            x = self.fuse(x)

        x = self.conv(x)
        if (not self.is_up) and (not self.is_down):
            return x
        elif self.is_up and (not self.is_down):
            x_up = self.up(x)
            return x, x_up
        elif (not self.is_up) and self.is_down:
            x_down = self.down(x)
            return x, x_down

        x_up = self.up(x)
        x_down = self.down(x)
        return x, x_up, x_down


class FR_UNet(nn.Module):
    def __init__(self,
                 num_classes=1,
                 num_channels=1,
                 feature_scale=2,
                 dropout=0.2,
                 fuse=True,
                 out_average=True,
                 ):
        super(FR_UNet, self).__init__()
        self.out_average = out_average
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / feature_scale) for x in filters]

        self.block1_3 = Block(
            num_channels, filters[0],  dp=dropout, is_up=False, is_down=True, fuse=fuse)
        self.block1_2 = Block(
            filters[0], filters[0],  dp=dropout, is_up=False, is_down=True, fuse=fuse)
        self.block1_1 = Block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=True, fuse=fuse)

        self.block10 = Block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=True, fuse=fuse)
        self.block11 = Block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=True, fuse=fuse)
        self.block12 = Block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=False, fuse=fuse)
        self.block13 = Block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=False, fuse=fuse)

        self.block2_2 = Block(
            filters[1], filters[1],  dp=dropout, is_up=True, is_down=True, fuse=fuse)
        self.block2_1 = Block(
            filters[1]*2, filters[1],  dp=dropout, is_up=True, is_down=True, fuse=fuse)

        self.block20 = Block(
            filters[1]*3, filters[1],  dp=dropout, is_up=True, is_down=True, fuse=fuse)
        self.block21 = Block(
            filters[1]*3, filters[1],  dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.block22 = Block(
            filters[1]*3, filters[1],  dp=dropout, is_up=True, is_down=False, fuse=fuse)

        self.block3_1 = Block(
            filters[2], filters[2],  dp=dropout, is_up=True, is_down=True, fuse=fuse)
        self.block30 = Block(
            filters[2]*2, filters[2],  dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.block31 = Block(
            filters[2]*3, filters[2],  dp=dropout, is_up=True, is_down=False, fuse=fuse)

        self.block40 = Block(
            filters[3], filters[3],
            dp=dropout, is_up=True, is_down=False, fuse=fuse)

        self.final1 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final2 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final3 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final4 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final5 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.fuse = nn.Conv2d(
            5, num_classes, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        x1_3, x_down1_3 = self.block1_3(x)
        x1_2, x_down1_2 = self.block1_2(x1_3)
        x2_2, x_up2_2, x_down2_2 = self.block2_2(x_down1_3)

        x1_1, x_down1_1 = self.block1_1(torch.cat([x1_2, x_up2_2], dim=1))
        x2_1, x_up2_1, x_down2_1 = self.block2_1(
            torch.cat([x_down1_2, x2_2], dim=1))

        x3_1, x_up3_1, x_down3_1 = self.block3_1(x_down2_2)
        x10, x_down10 = self.block10(torch.cat([x1_1, x_up2_1], dim=1))
        x20, x_up20, x_down20 = self.block20(
            torch.cat([x_down1_1, x2_1, x_up3_1], dim=1))
        x30, x_up30 = self.block30(torch.cat([x_down2_1, x3_1], dim=1))

        _, x_up40 = self.block40(x_down3_1)
        x11, x_down11 = self.block11(torch.cat([x10, x_up20], dim=1))
        x21, x_up21 = self.block21(torch.cat([x_down10, x20, x_up30], dim=1))
        _, x_up31 = self.block31(torch.cat([x_down20, x30, x_up40], dim=1))
        x12 = self.block12(torch.cat([x11, x_up21], dim=1))
        _, x_up22 = self.block22(torch.cat([x_down11, x21, x_up31], dim=1))
        x13 = self.block13(torch.cat([x12, x_up22], dim=1))

        if self.out_average:
            output = (
                self.final1(x1_1)
                + self.final2(x10)
                + self.final3(x11)
                + self.final4(x12)
                + self.final5(x13)
            ) / 5
        else:
            output = self.final5(x13)

        return output


def check():
    model = FR_UNet(num_channels=3, num_classes=1, fuse=True, out_average=True)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    out = model(torch.rand(2, 3, 48, 48))
    print(out.shape, out.min(), out.max())


if __name__ == "__main__":
    check()
