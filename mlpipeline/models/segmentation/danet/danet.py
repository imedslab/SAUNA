import math
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from mlpipeline.models.segmentation.danet.backbone import VGG, Conv2d_BN, resnet50
from mlpipeline.models.segmentation.danet.vit import ViT, CrossTransformer


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.convs = nn.ModuleList([
            Conv2d_BN(in_channels, in_channels,
                      kernel_size=(3, 1), bias=False),
            Conv2d_BN(in_channels, in_channels,
                      kernel_size=(1, 3), bias=False),
            Conv2d_BN(in_channels, in_channels, kernel_size=3,
                      bias=False, strip_type="upward"),
            Conv2d_BN(in_channels, in_channels, kernel_size=3,
                      bias=False, strip_type="downward"),
        ])
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=in_channels, out_features=in_channels)

        self.up = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True)
        self.conv1 = Conv2d_BN(in_channels, out_channels, 1)

    def forward(self, inputs):
        xs = [self.relu(conv(inputs)) for conv in self.convs]
        ps = [self.avg_pool(x).squeeze(dim=3).squeeze(dim=2) for x in xs]
        ws = torch.cat([self.fc(p).unsqueeze(dim=-1) for p in ps], dim=-1)
        ws = self.softmax(ws).unsqueeze(dim=2).unsqueeze(dim=3)
        xs = torch.stack(xs, dim=-1)
        x = torch.sum(ws * xs, dim=-1)

        x = self.up(x)
        x = self.relu(self.conv1(x))
        return x


class DANet(nn.Module):
    def __init__(self, in_channels, out_channels, num_patches, dropout=0.0):
        super(DANet, self).__init__()
        self.num_patches = num_patches
        self.num_rc = int(math.sqrt(num_patches))
        features_dim_list = [32, 64, 96, 128, 196]
        self.backbone = VGG(in_channels, features_dim_list)
        # self.backbone = resnet50()

        self.transformer_patch = ViT(
            num_inputs=num_patches,
            image_size=10,
            patch_size=1,
            channels=196,
            dim=196,
            depth=1,
            heads=4,
            mlp_dim=196,
            dim_head=64,
            dropout=dropout,
        )
        self.transformer_whole = ViT(
            num_inputs=1,
            image_size=10,
            patch_size=1,
            channels=196,
            dim=196,
            depth=1,
            heads=4,
            mlp_dim=196,
            dim_head=64,
            dropout=dropout,
        )
        self.transformer_cross = CrossTransformer(
            dim=196,
            depth=1,
            heads=4,
            dim_head=64,
            mlp_dim=196,
            dropout=dropout,
        )

        self.up4 = UpBlock(features_dim_list[4], features_dim_list[3])
        self.up3 = UpBlock(features_dim_list[3], features_dim_list[2])
        self.up2 = UpBlock(features_dim_list[2], features_dim_list[1])
        self.up1 = UpBlock(features_dim_list[1], features_dim_list[0])
        self.conv_out = nn.Conv2d(features_dim_list[0], out_channels, 1)

    def forward(self, inputs):
        b, n, _, _, _ = inputs.shape
        bp = b * (n - 1)

        x = Rearrange("b n c h w -> (b n) c h w")(inputs)
        x1, x2, x3, x4, x5 = self.backbone(x)

        f = Rearrange("(b n) c h w -> b n c h w", b=inputs.shape[0])(x5)
        f_patch = f[:, :-1, ...]
        f_whole = f[:, -1:, ...]

        f_patch = self.transformer_patch(f_patch)
        f_whole = self.transformer_whole(f_whole)
        f_patch = self.transformer_cross(f_patch, f_whole)

        y5 = Rearrange("b (n h w) c -> (b n) c h w",
                       h=x5.shape[2], w=x5.shape[3])(f_patch)
        y4 = self.up4(y5)
        y3 = self.up3(y4 + x4[:bp, ...])
        y2 = self.up2(y3 + x3[:bp, ...])
        y1 = self.up1(y2 + x2[:bp, ...])
        y = y1 + x1[:bp, ...]

        out = self.conv_out(y)

        # out = Rearrange("(b p1 p2) c h w -> b c (h p1) (w p2)", p1=size, p2=size)(out)
        out = Rearrange("(b n) c h w -> b n c h w", n=self.num_patches)(out)
        out = torch.split(out, split_size_or_sections=self.num_rc, dim=1)
        out = torch.cat(out, dim=3)
        out = torch.split(out, split_size_or_sections=1, dim=1)
        out = torch.cat(out, dim=4)
        out = out.squeeze(dim=1)
        return out


def check():
    device = torch.device("cpu")

    model = DANet(in_channels=3, out_channels=1, num_patches=4)
    model.to(device)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    out = model(torch.rand(2, 5, 3, 160, 160).to(device))
    print(out.shape, out.min(), out.max())
    loss = nn.L1Loss()(out, torch.rand(2, 1, 320, 320).to(device))
    loss.backward()

    print(model.up1.convs[2].conv.weight[0, 0, :, :])
    print(model.up1.convs[3].conv.weight[5, 5, :, :])


if __name__ == "__main__":
    check()
