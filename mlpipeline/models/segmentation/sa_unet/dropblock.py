import torch
import torch.nn as nn


class DropBlock2d(nn.Module):
    def __init__(self, block_size, keep_prob, sync_channels=False):
        super(DropBlock2d, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.sync_channels = sync_channels

        self.max_pool = nn.MaxPool2d(
            kernel_size=(self.block_size, self.block_size),
            padding=(self.block_size - 1) // 2,
            stride=1,
        )

    def _get_gamma(self, height, width):
        height = torch.tensor(height, dtype=torch.float32)
        width = torch.tensor(width, dtype=torch.float32)
        block_size = torch.tensor(self.block_size, dtype=torch.float32)
        return ((1.0 - self.keep_prob) / (block_size ** 2)) *\
            (height * width / ((height - block_size + 1.0) * (width - block_size + 1.0)))

    def _compute_valid_seed_region(self, height, width):
        positions = torch.cat([
            torch.arange(start=0, end=height).unsqueeze(
                dim=1).repeat([1, width]).unsqueeze(dim=-1),
            torch.arange(start=0, end=width).unsqueeze(
                dim=0).repeat([height, 1]).unsqueeze(dim=-1),
        ], dim=-1)

        half_block_size = self.block_size // 2
        valid_seed_region = torch.where(
            torch.all(
                torch.stack(
                    [
                        positions[:, :, 0] >= half_block_size,
                        positions[:, :, 1] >= half_block_size,
                        positions[:, :, 0] < height - half_block_size,
                        positions[:, :, 1] < width - half_block_size,
                    ],
                    dim=-1,
                ),
                dim=-1,
            ),
            torch.ones((height, width)),
            torch.zeros((height, width)),
        )
        return valid_seed_region.unsqueeze(dim=0).unsqueeze(dim=-1)

    def _compute_drop_mask(self, shape):
        height, width = shape[1], shape[2]
        mask = torch.bernoulli(torch.full(
            shape, fill_value=self._get_gamma(height, width)))
        mask *= self._compute_valid_seed_region(height, width)

        mask = self.max_pool(mask)
        return 1.0 - mask

    def forward(self, inputs):
        def dropped_inputs():
            outputs = inputs
            outputs = outputs.permute([0, 2, 3, 1])

            shape = outputs.shape
            if self.sync_channels:
                mask = self._compute_drop_mask(
                    [shape[0], shape[1], shape[2], 1])
            else:
                mask = self._compute_drop_mask(shape)

            mask = mask.to(outputs.device)
            weights = torch.prod(torch.tensor(shape)).float() / torch.sum(mask)
            outputs = outputs * mask * weights

            outputs = outputs.permute([0, 3, 1, 2])
            return outputs

        return dropped_inputs() if self.training else inputs
