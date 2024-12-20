import torch
import torch.nn as nn
import torch.nn.functional as functional

from mlpipeline.models.segmentation.sa_unet.dropblock import DropBlock2d
from mlpipeline.models.segmentation.sa_unet.attention import SpatialAttention


class Block(nn.Module):
    def __init__(
        self,
        in_channels, out_channels,
        kernel_size, block_size,
        keep_prob,
        spatial_attention=False,
    ):
        super(Block, self).__init__()
        padding = (kernel_size - 1) // 2
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size, stride=1, padding=padding)
        self.db1 = DropBlock2d(block_size, keep_prob)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.attention = SpatialAttention() if spatial_attention else nn.Identity()

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size, stride=1, padding=padding)
        self.db2 = DropBlock2d(block_size, keep_prob)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, inputs):
        x = self.bn1(self.db1(self.conv1(inputs)))
        x = self.relu(x)
        x = self.attention(x)
        x = self.bn2(self.db2(self.conv2(x)))
        x = self.relu(x)
        return x


class SA_Unet(nn.Module):
    def __init__(
        self,
        num_channels, num_classes,
        block_size=7, keep_prob=0.82,
        start_neurons=16,
    ):
        super(SA_Unet, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2)

        self.block1 = Block(
            in_channels=num_channels, out_channels=start_neurons,
            kernel_size=3, block_size=block_size,
            keep_prob=keep_prob,
        )
        self.block2 = Block(
            in_channels=start_neurons, out_channels=start_neurons * 2,
            kernel_size=3, block_size=block_size,
            keep_prob=keep_prob,
        )
        self.block3 = Block(
            in_channels=start_neurons * 2, out_channels=start_neurons * 4,
            kernel_size=3, block_size=block_size,
            keep_prob=keep_prob,
        )
        self.block4 = Block(
            in_channels=start_neurons * 4, out_channels=start_neurons * 8,
            kernel_size=3, block_size=block_size,
            keep_prob=keep_prob, spatial_attention=True,
        )

        self.deconv5 = nn.ConvTranspose2d(
            in_channels=start_neurons * 8, out_channels=start_neurons * 4,
            kernel_size=3, stride=2, padding=1, output_padding=1,
        )
        self.block5 = Block(
            in_channels=start_neurons * 8, out_channels=start_neurons * 4,
            kernel_size=3, block_size=block_size,
            keep_prob=keep_prob,
        )

        self.deconv6 = nn.ConvTranspose2d(
            in_channels=start_neurons * 4, out_channels=start_neurons * 2,
            kernel_size=3, stride=2, padding=1, output_padding=1,
        )
        self.block6 = Block(
            in_channels=start_neurons * 4, out_channels=start_neurons * 2,
            kernel_size=3, block_size=block_size,
            keep_prob=keep_prob,
        )

        self.deconv7 = nn.ConvTranspose2d(
            in_channels=start_neurons * 2, out_channels=start_neurons,
            kernel_size=3, stride=2, padding=1, output_padding=1,
        )
        self.block7 = Block(
            in_channels=start_neurons * 2, out_channels=start_neurons,
            kernel_size=3, block_size=block_size,
            keep_prob=keep_prob,
        )

        self.conv_out = nn.Conv2d(
            start_neurons, num_classes, kernel_size=1, padding=0)

    def forward(self, inputs):
        x1 = self.block1(inputs)
        p1 = self.max_pool(x1)

        x2 = self.block2(p1)
        p2 = self.max_pool(x2)

        x3 = self.block3(p2)
        p3 = self.max_pool(x3)

        x4 = self.block4(p3)

        x5 = self.deconv5(x4)
        x5 = torch.cat([x5, x3], dim=1)
        x5 = self.block5(x5)

        x6 = self.deconv6(x5)
        x6 = torch.cat([x6, x2], dim=1)
        x6 = self.block6(x6)

        x7 = self.deconv7(x6)
        x7 = torch.cat([x7, x1], dim=1)
        x7 = self.block7(x7)

        out = self.conv_out(x7)
        return out


def check():
    device = torch.device("cuda:0")
    model = SA_Unet(num_channels=3, num_classes=1, start_neurons=16).to(device)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    model.train()
    out = model(torch.rand(2, 3, 592, 592).to(device))
    print(out.shape, out.min(), out.max())


if __name__ == "__main__":
    check()
