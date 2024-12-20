import torch
import torch.nn as nn

import mlpipeline.models.segmentation.magfnet.base as base


class MsaBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(MsaBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = base.Conv2d_BN(in_channels, out_channels, kernel_size)
        self.conv2 = base.Conv2d_BN(out_channels, out_channels, kernel_size)
        self.conv3 = base.Conv2d_BN(
            out_channels * 2, out_channels, kernel_size)

        self.se = base.SqueezeExcitation(
            out_channels * 3, out_channels, activation=self.relu)
        self.conv4 = base.Conv2d_BN(out_channels * 3, out_channels, 1)

    def forward(self, inputs):
        x_a = self.relu(self.conv1(inputs))
        x_b = self.relu(self.conv2(x_a))
        x_ab = self.relu(self.conv3(torch.cat([x_a, x_b], dim=1)))
        x_c = torch.cat([x_ab, x_a, x_b], dim=1)

        x = self.se(x_c)
        x = self.relu(self.conv4(x))
        return x


class HfpBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=2):
        super(HfpBlock, self).__init__()
        padding = (kernel_size - 1) // 2

        self.max_pool = nn.MaxPool2d(
            kernel_size=kernel_size, stride=stride, padding=padding)
        self.avg_pool = nn.AvgPool2d(
            kernel_size=kernel_size, stride=stride, padding=padding)

        self.conv1 = base.Conv2d_BN(
            in_channels, in_channels, kernel_size, stride=stride)
        self.relu = nn.ReLU(inplace=True)

        self.se = base.SqueezeExcitation(
            in_channels * 3, in_channels, activation=self.relu)
        self.conv2 = base.Conv2d_BN(in_channels * 3, in_channels, 1)

    def forward(self, inputs):
        m = self.max_pool(inputs)
        a = self.avg_pool(inputs)
        c = self.relu(self.conv1(inputs))

        x = self.se(torch.cat([m, a, c], dim=1))
        x = self.relu(self.conv2(x))
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, features_dim, stride=2):
        super(Encoder, self).__init__()

        self.block1 = MsaBlock(in_channels, features_dim, kernel_size=3)
        self.pool1 = HfpBlock(features_dim, kernel_size=3, stride=stride)

        self.block2 = MsaBlock(features_dim, features_dim * 2, kernel_size=3)
        self.pool2 = HfpBlock(features_dim * 2, kernel_size=3, stride=stride)

        self.block3 = MsaBlock(
            features_dim * 2, features_dim * 4, kernel_size=3)
        self.pool3 = HfpBlock(features_dim * 4, kernel_size=3, stride=stride)

        self.block4 = MsaBlock(
            features_dim * 4, features_dim * 8, kernel_size=3)

    def forward(self, inputs):
        x1 = self.block1(inputs)
        x2 = self.block2(self.pool1(x1))
        x3 = self.block3(self.pool2(x2))
        x4 = self.block4(self.pool3(x3))
        return (x1, x2, x3, x4)


class Decoder(nn.Module):
    def __init__(self, features_dim, out_channels, stride=2):
        super(Decoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.up41 = nn.ConvTranspose2d(
            features_dim * 8, features_dim * 4, kernel_size=2, stride=stride)
        self.up42 = nn.ConvTranspose2d(
            features_dim * 8, features_dim * 4, kernel_size=2, stride=stride)
        self.block3 = MsaBlock(
            features_dim * 4, features_dim * 4, kernel_size=3)

        self.up31 = nn.ConvTranspose2d(
            features_dim * 4, features_dim * 2, kernel_size=2, stride=stride)
        self.up32 = nn.ConvTranspose2d(
            features_dim * 4, features_dim * 2, kernel_size=2, stride=stride)
        self.block2 = MsaBlock(
            features_dim * 2, features_dim * 2, kernel_size=3)

        self.up21 = nn.ConvTranspose2d(
            features_dim * 2, features_dim, kernel_size=2, stride=stride)
        self.up22 = nn.ConvTranspose2d(
            features_dim * 2, features_dim, kernel_size=2, stride=stride)
        self.block1 = MsaBlock(features_dim, features_dim, kernel_size=3)

        self.conv_out = nn.Conv2d(features_dim, out_channels, kernel_size=1)

    def forward(self, x4, x4f, x3f, x2f):
        x4 = self.up41(x4)
        x4f = self.up42(x4f)
        x3 = x4 + x4f
        x3 = self.block3(x3)

        x3 = self.up31(x3)
        x3f = self.up32(x3f)
        x2 = x3 + x3f
        x2 = self.block2(x2)

        x2 = self.up21(x2)
        x2f = self.up22(x2f)
        x1 = x2 + x2f
        x1 = self.block1(x1)

        out = self.conv_out(x1)
        return out
