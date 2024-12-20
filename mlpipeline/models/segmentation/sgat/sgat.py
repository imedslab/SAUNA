import torch
import torch.nn as nn

from mlpipeline.models.segmentation.sgat import res_modules
from mlpipeline.models.segmentation.sgat import vit


class MixedEncoder(nn.Module):
    def __init__(self, base_width, patch_dim, patch_size, num_patches, dropout):
        super(MixedEncoder, self).__init__()

        self.res_encoder = res_modules.ResEncoder(
            inplanes=base_width, num_blocks=1, width=base_width, stride=1)

        self.pe1 = vit.PatchEmbedding(
            base_width, patch_dim, patch_size, num_patches)
        self.transformer = vit.TransformerEncoder(
            patch_dim, patch_dim, dropout=0.0)

        self.pe2 = vit.PatchEmbedding(
            base_width * 2, patch_dim, patch_size, num_patches)
        self.sgaff = vit.SGAFF(patch_dim, patch_dim, patch_size, dropout)

    def forward(self, x):
        res_out = self.res_encoder(x)

        patches_vit = self.pe1(x)
        vit_out = self.transformer(patches_vit)

        patches_res = self.pe2(res_out)
        out = self.sgaff(patches_res, vit_out)
        return out


class SGAT(nn.Module):
    def __init__(self, in_channels, num_classes, base_width, num_patches, dropout=0.0):
        super(SGAT, self).__init__()

        # Down Path
        self.first_down = res_modules.FirstDown(
            in_channels=in_channels, out_channels=base_width)

        # in 32, base 32, expand 64
        self.encoder1_1 = MixedEncoder(
            base_width, base_width * 2,
            8, num_patches,
            dropout)
        self.encoder1_2 = MixedEncoder(
            base_width * 2, base_width * 2,
            8, num_patches,
            dropout)
        self.encoder1_3 = MixedEncoder(
            base_width * 2, base_width * 2,
            8, num_patches,
            dropout)

        # in 64, base 64, expand 128
        base_width *= 2
        self.res_encoder2 = res_modules.ResEncoder(
            inplanes=base_width, num_blocks=4, width=base_width, stride=2)

        # in 128, base 128, expand 256
        base_width *= 2
        self.res_encoder3 = res_modules.ResEncoder(
            inplanes=base_width, num_blocks=6, width=base_width, stride=2)

        # in 256, base 256, expand 512
        base_width *= 2
        self.res_encoder4 = res_modules.ResEncoder(
            inplanes=base_width, num_blocks=3, width=base_width, stride=2)

        # Up Path
        self.decoder1 = res_modules.UpBlock(
            in_channels=base_width * 2, out_channels=base_width)
        base_width //= 2

        self.decoder2 = res_modules.UpBlock(
            in_channels=base_width * 2, out_channels=base_width)
        base_width //= 2

        self.decoder3 = res_modules.UpBlock(
            in_channels=base_width * 2, out_channels=base_width)
        base_width //= 2

        self.decoder4 = res_modules.UpBlock(
            in_channels=base_width * 2, out_channels=base_width)
        self.last_up = res_modules.LastUp(
            in_channels=base_width, out_channels=base_width)

        # Output
        self.conv_out = nn.Conv2d(base_width, num_classes, 1)

    def forward(self, inputs):
        # Down Path
        down1, x = self.first_down(inputs)

        x = self.encoder1_1(x)
        x = self.encoder1_2(x)
        down2 = x = self.encoder1_3(x)

        down3 = x = self.res_encoder2(x)
        down4 = x = self.res_encoder3(x)
        x = self.res_encoder4(x)

        # up Path
        x = self.decoder1(x, down4)
        x = self.decoder2(x, down3)
        x = self.decoder3(x, down2)
        x = self.decoder4(x, down1)

        out = self.last_up(x)
        out = self.conv_out(out)
        return out


def check():
    model = SGAT(3, 1, 32, 16 * 16)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    out = model(torch.rand(2, 3, 512, 512))
    print(out.shape, out.min(), out.max())


if __name__ == "__main__":
    check()
