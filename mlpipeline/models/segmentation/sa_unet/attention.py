import torch
import torch.nn as nn
from einops.layers.torch import Reduce


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.max_pool = Reduce("b c h w -> b 1 h w", "max")
        self.avg_pool = Reduce("b c h w -> b 1 h w", "mean")

    def forward(self, inputs):
        max_pooled = self.max_pool(inputs)
        avg_pooled = self.avg_pool(inputs)

        x = torch.cat([max_pooled, avg_pooled], dim=1)
        x = self.conv(x)
        weights = torch.sigmoid(x)

        return (weights * inputs)
