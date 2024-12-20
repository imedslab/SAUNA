import math

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_dim, patch_size, num_patches):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=patch_size,
            stride=patch_size,
            groups=in_channels,
            padding=0,
        )

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c h w -> b (h w) c"),
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, patch_dim),
            nn.LayerNorm(patch_dim),
        )
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches, patch_dim))

    def forward(self, x):
        assert x.shape[2] % self.patch_size == 0

        x = self.conv(x)
        x = self.to_patch_embedding(x)
        _, n, _ = x.shape

        x += self.pos_embedding[:, :(n + 1)]
        return x


class Attention(nn.Module):
    def __init__(self, in_channels, dim, dropout=0.0):
        super(Attention, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(in_channels * 2, dim * 2)
        self.ln1 = nn.LayerNorm(dim * 2)

        self.fc2_a = nn.Linear(dim * 2, dim)
        self.ln2_a = nn.LayerNorm(dim)
        self.fc2_m = nn.Linear(dim * 2, dim)
        self.ln2_m = nn.LayerNorm(dim)

        self.softmax = nn.Softmax(dim=-1)
        self.ln3 = nn.LayerNorm(dim)

    def forward(self, x, y):
        q_es = torch.stack([x, y], dim=2).permute(0, 3, 4, 1, 2)
        q_es = q_es.unsqueeze(dim=-2)

        out = torch.cat([x, y], dim=1)
        out = self.global_pool(out).squeeze(dim=3).squeeze(dim=2)

        out = self.ln1(self.fc1(out))
        out_ba = self.ln2_a(self.fc2_a(out))
        out_bm = self.ln2_m(self.fc2_m(out))

        out = torch.stack([out_ba, out_bm], dim=-1)
        attn = self.softmax(out)
        attn = attn.view(q_es.shape[0], 1, 1, q_es.shape[3], 2, 1)

        # [B, H, W, C, 1, 2] @ [B, 1, 1, C, 2, 1] => [B, H, W, C, 1, 1] => [B, C, H, W]
        out = torch.matmul(q_es, attn)
        out = out.squeeze(dim=5).squeeze(dim=4)
        out = self.ln3(out).permute(0, 3, 1, 2)
        return out


class SGAP(nn.Module):
    def __init__(self, in_channels, dim, dropout=0.0):
        super(SGAP, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3,
                      padding=1, groups=in_channels),
            nn.Conv2d(in_channels, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
        )
        self.ap = nn.AvgPool2d(kernel_size=3, padding=1, stride=1)
        self.mp = nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
        self.attn = Attention(dim, dim, dropout)

    def forward(self, x):
        num_patches = x.shape[1]
        rows = cols = int(math.sqrt(num_patches))

        x = Rearrange("b (h w) c -> b c h w", h=cols, w=rows)(x)

        x = self.relu(self.conv(x))
        x = self.dropout(x)
        xap = self.ap(x)
        xmp = self.mp(x)

        out = self.attn(xap, xmp)
        out = Rearrange("b c h w -> b (h w) c")(out)
        return out


class MHA_SGAP(nn.Module):
    def __init__(self, in_channels, dim, dropout=0.0):
        super(MHA_SGAP, self).__init__()
        self.dim = dim
        self.scale = dim ** -0.5
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(in_channels, dim * 3)
        self.ln1 = nn.LayerNorm(dim * 3)

        self.sgap_q = SGAP(dim, dim, dropout)
        self.sgap_k = SGAP(dim, dim, dropout)
        self.sgap_v = SGAP(dim, dim, dropout)
        self.attend = nn.Softmax(dim=-1)

        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.ln1(self.fc1(x))
        q, k, v = torch.chunk(x, 3, dim=-1)

        q = self.sgap_q(q)
        k = self.sgap_k(k)
        v = self.sgap_v(v)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)

        out = self.fc2(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super(FeedForward, self).__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.mlp(x)
        out = self.dropout(out)
        out = out + x
        return out


class TransformerEncoder(nn.Module):
    def __init__(self, dim, mlp_dim, dropout=0.0):
        super(TransformerEncoder, self).__init__()

        self.norm = nn.LayerNorm(dim)
        self.attn = MHA_SGAP(dim, dim, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, x):
        x = self.norm(x)
        x = self.attn(x) + x

        x = self.mlp(x)
        return x


class SGAFF(nn.Module):
    def __init__(self, in_channels, dim, patch_size, dropout=0.0):
        super(SGAFF, self).__init__()
        self.patch_size = patch_size
        self.dropout = nn.Dropout(dropout)
        self.attn = Attention(in_channels, dim, dropout)

        self.fc = nn.Linear(dim, dim * patch_size * patch_size)
        self.ln = nn.LayerNorm(dim * patch_size * patch_size)

        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim))
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        assert x.shape[1] == y.shape[1]
        num_patches = x.shape[1]
        rows = cols = int(math.sqrt(num_patches))

        x = Rearrange("b (h w) c -> b c h w", h=cols, w=rows)(x)
        y = Rearrange("b (h w) c -> b c h w", h=cols, w=rows)(y)

        out = self.attn(x, y)
        out = Rearrange("b c h w -> b h w c")(out)
        size = out.shape[1] * self.patch_size

        out = Rearrange("b h w c -> b (h w) c")(out)
        out = self.ln(self.fc(out))
        out = Rearrange(
            "b (h w) (c p1 p2) -> b h p1 w p2 c",
            h=size // self.patch_size,
            w=size // self.patch_size,
            p1=self.patch_size,
            p2=self.patch_size,
        )(out)
        # out = out.permute(0, 1, 4, 2, 5, 3)
        out = Rearrange("b h p1 w p2 c -> b c (h p1) (w p2)")(out)

        out = self.relu(self.conv1(out))
        out = self.dropout(out)
        out = self.relu(self.conv2(out))
        return out
