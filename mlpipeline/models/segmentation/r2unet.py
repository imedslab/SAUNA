import torch
import torch.nn as nn


class RC_block(nn.Module):
    def __init__(self, channel, t=2):
        super().__init__()
        self.t = t

        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        r_x = self.conv(x)

        for _ in range(self.t):
            r_x = self.conv(x + r_x)

        return r_x


class RRC_block(nn.Module):
    def __init__(self, channel, t=2):
        super().__init__()

        self.RC_net = nn.Sequential(
            RC_block(channel, t=t),
            RC_block(channel, t=t),
        )

    def forward(self, x):
        res_x = self.RC_net(x)
        return x + res_x


class R2UNet(nn.Module):
    def __init__(self, num_channels=3, num_filters=64, num_classes=1):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels, num_filters, 3, 1, 1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(True),
            RRC_block(num_filters),
        )

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(num_filters, num_filters * 2, 3, 1, 1),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU(True),
            RRC_block(num_filters * 2),
        )

        self.conv3 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(num_filters * 2, num_filters * 4, 3, 1, 1),
            nn.BatchNorm2d(num_filters * 4),
            nn.ReLU(True),
            RRC_block(num_filters * 4),
        )

        self.conv4 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(num_filters * 4, num_filters * 8, 3, 1, 1),
            nn.BatchNorm2d(num_filters * 8),
            nn.ReLU(True),
            RRC_block(num_filters * 8),
        )

        self.trans_conv = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(num_filters * 8, num_filters * 16, 3, 1, 1),
            nn.BatchNorm2d(num_filters * 16),
            nn.ReLU(True),
            RRC_block(num_filters * 16),
            nn.ConvTranspose2d(num_filters * 16, num_filters *
                               8, kernel_size=2, stride=2),
        )

        self.up_conv1 = nn.Sequential(
            nn.Conv2d(num_filters * 16, num_filters * 8, 3, 1, 1),
            nn.BatchNorm2d(num_filters * 8),
            nn.ReLU(True),
            RRC_block(num_filters * 8),
            nn.ConvTranspose2d(num_filters * 8, num_filters *
                               4, kernel_size=2, stride=2),
        )

        self.up_conv2 = nn.Sequential(
            nn.Conv2d(num_filters * 8, num_filters * 4, 3, 1, 1),
            nn.BatchNorm2d(num_filters * 4),
            nn.ReLU(True),
            RRC_block(num_filters * 4),
            nn.ConvTranspose2d(num_filters * 4, num_filters *
                               2, kernel_size=2, stride=2),
        )

        self.up_conv3 = nn.Sequential(
            nn.Conv2d(num_filters * 4, num_filters * 2, 3, 1, 1),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU(True),
            RRC_block(num_filters * 2),
            nn.ConvTranspose2d(num_filters * 2, num_filters,
                               kernel_size=2, stride=2),
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(num_filters * 2, num_filters, 3, 1, 1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(True),
            RRC_block(num_filters),
            nn.Conv2d(num_filters, num_classes, 1),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        x = self.trans_conv(x4)

        x = self.up_conv1(torch.cat((x, x4), dim=1))
        x = self.up_conv2(torch.cat((x, x3), dim=1))
        x = self.up_conv3(torch.cat((x, x2), dim=1))
        x = self.final_conv(torch.cat((x, x1), dim=1))

        return x


def check():
    model = R2UNet(3, 16, 1)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    out = model(torch.rand(2, 3, 128, 128))
    print(out.shape, out.min(), out.max())


if __name__ == "__main__":
    check()
