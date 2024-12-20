import torch
import torch.nn as nn


class Conv2d_BN(nn.Module):
    def __init__(
        self,
        num_in_filters,
        num_out_filters,
        kernel_size,
        stride=(1, 1),
        use_bn=False,
    ):
        super().__init__()
        padding = (kernel_size[0] - 1) // 2
        self.conv = nn.Conv2d(
            in_channels=num_in_filters,
            out_channels=num_out_filters,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        if use_bn:
            self.batchnorm = nn.BatchNorm2d(num_out_filters)
        else:
            self.batchnorm = None
        return

    def forward(self, x):
        x = self.conv(x)
        if self.batchnorm is not None:
            x = self.batchnorm(x)
        return x


class DownBlock(nn.Module):
    def __init__(
        self, num_in_filters, num_out_filters, kernel_size,
        dropout, activation, max_pool,
    ):
        super().__init__()
        self.conv1 = Conv2d_BN(num_in_filters, num_out_filters, kernel_size)
        self.conv2 = Conv2d_BN(num_out_filters, num_out_filters, kernel_size)
        self.dropout = dropout
        self.activation = activation
        self.max_pool = max_pool

    def forward(self, x):
        out = self.dropout(self.activation(self.conv1(x)))
        out = self.dropout(self.activation(self.conv2(out)))
        pool = self.max_pool(out)
        return out, pool


class UpBlock(nn.Module):
    def __init__(
        self, num_in_filters, num_out_filters, kernel_size_up, kernel_size,
        dropout, activation,
    ):
        super().__init__()
        self.conv_up = nn.ConvTranspose2d(
            num_in_filters, num_out_filters, kernel_size_up,
            stride=(2, 2), padding=0,
        )
        # self.bn_up = nn.BatchNorm2d(num_out_filters)

        self.conv1 = Conv2d_BN(num_in_filters, num_out_filters, kernel_size)
        self.conv2 = Conv2d_BN(num_out_filters, num_out_filters, kernel_size)
        self.dropout = dropout
        self.activation = activation

    def forward(self, x, y):
        x = self.conv_up(x)
        x = torch.cat([x, y], dim=1)

        out = self.dropout(self.activation(self.conv1(x)))
        out = self.dropout(self.activation(self.conv2(out)))
        return out


class IterNet(nn.Module):
    def __init__(self, num_channels=3, num_filters=32, num_classes=1, dropout=0.0, activation="relu", iteration=1):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        if activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.1)
        else:
            self.activation = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.identity = nn.Identity()

        self.iteration = iteration

        # Down
        self.down1 = DownBlock(
            num_channels, num_filters, (3, 3),
            self.dropout, self.activation, self.max_pool)
        self.down2 = DownBlock(
            num_filters, num_filters * 2, (3, 3),
            self.dropout, self.activation, self.max_pool)
        self.down3 = DownBlock(
            num_filters * 2, num_filters * 4, (3, 3),
            self.dropout, self.activation, self.max_pool)
        self.down4 = DownBlock(
            num_filters * 4, num_filters * 8, (3, 3),
            self.dropout, self.activation, self.max_pool)
        self.down5 = DownBlock(
            num_filters * 8, num_filters * 16, (3, 3),
            self.dropout, self.activation, self.identity)

        # Up
        self.up6 = UpBlock(
            num_filters * 16, num_filters * 8, (2, 2), (3, 3),
            self.dropout, self.activation)
        self.up7 = UpBlock(
            num_filters * 8, num_filters * 4, (2, 2), (3, 3),
            self.dropout, self.activation)
        self.up8 = UpBlock(
            num_filters * 4, num_filters * 2, (2, 2), (3, 3),
            self.dropout, self.activation)
        self.up9 = UpBlock(
            num_filters * 2, num_filters, (2, 2), (3, 3),
            self.dropout, self.activation)

        # PT
        self.pt1 = DownBlock(
            num_filters, num_filters, (3, 3),
            self.dropout, self.activation, self.identity)
        self.pt2 = DownBlock(
            num_filters, num_filters * 2, (3, 3),
            self.dropout, self.activation, self.max_pool)
        self.pt3 = DownBlock(
            num_filters * 2, num_filters * 4, (3, 3),
            self.dropout, self.activation, self.identity)
        self.pt8 = UpBlock(
            num_filters * 4, num_filters * 2, (2, 2), (3, 3),
            self.dropout, self.activation)
        self.pt9 = UpBlock(
            num_filters * 2, num_filters, (2, 2), (3, 3),
            self.dropout, self.activation)

        self.convs_a = nn.ModuleList([
            Conv2d_BN(num_filters * (2 + i), num_filters, (1, 1), 1)
            for i in range(self.iteration)
        ])
        self.convs_out = nn.ModuleList([
            nn.Conv2d(num_filters, num_classes, (1, 1), 1, 0)
            for i in range(self.iteration + 1)
        ])

    def forward(self, x):
        # Down
        conv1, pool1 = self.down1(x)
        a = conv1
        conv2, pool2 = self.down2(pool1)
        conv3, pool3 = self.down3(pool2)
        conv4, pool4 = self.down4(pool3)
        conv5, _ = self.down5(pool4)

        # Up
        conv6 = self.up6(conv5, conv4)
        conv7 = self.up7(conv6, conv3)
        conv8 = self.up8(conv7, conv2)
        conv9 = self.up9(conv8, conv1)

        # Refine
        conv9s = [conv9]
        outs = []
        a_layers = [a]

        for iteration_id in range(self.iteration):
            out = self.convs_out[iteration_id](conv9s[-1])
            outs.append(out)

            conv1, _ = self.pt1(conv9s[-1])
            a_layers.append(conv1)
            conv1 = torch.cat(a_layers, dim=1)
            conv1 = self.convs_a[iteration_id](conv1)
            pool1 = self.max_pool(conv1)

            conv2, pool2 = self.pt2(pool1)
            conv3, _ = self.pt3(pool2)

            conv8 = self.pt8(conv3, conv2)
            conv9 = self.pt9(conv8, conv1)

        out = self.convs_out[self.iteration](conv9s[-1])
        outs.append(out)

        return outs


def check():
    model = IterNet(3, 32, 1, 0.1, "relu", 3)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    outs = model(torch.rand(2, 3, 128, 128))
    print([out.shape for out in outs])


if __name__ == "__main__":
    check()
