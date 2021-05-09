"""
paper - https://arxiv.org/pdf/1505.04597.pdf
"""

import torch
from torch import nn
from torchvision.transforms import functional as TF


class Unet(nn.Module):
    def __init__(self, in_channels: int, n_classes: int):
        """

        Args:
            in_channels
            n_classes

        """
        super(Unet, self).__init__()

        self.in_channels = in_channels
        self.n_classes = n_classes

        # TODO: Generalize channel list
        self.sizes = [64, 128, 256, 512]

        self.convdown = nn.ModuleList()
        self.convup = nn.ModuleList()

        for size in self.sizes:
            self.convdown.append(self.DoubleConv(in_channels, size))
            in_channels = size

        for size in reversed(self.sizes):
            self.convup.append(self.upsample(size * 2, size))
            self.convup.append(self.DoubleConv(size * 2, size))

        self.bottleNeck = self.DoubleConv(self.sizes[-1], self.sizes[-1] * 2)
        self.convOut = nn.Conv2d(self.sizes[0], self.n_classes, kernel_size=(1, 1))

    @staticmethod
    def DoubleConv(in_channels: int, out_channels: int):
        """
        Args:
            in_channels
            out_channels

        Returns:
            double conv layers

        """
        doubleconv: nn.Sequential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        return doubleconv

    @staticmethod
    def upsample(in_channels: int, out_channels: int):
        """
            Args:
                in_channels
                out_channels

            Returns:
                Trainable upsample layer. (convTranspose)
            """
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2, 2), padding=1, stride=1)

    def forward(self, x):
        skip_connections = []

        for downsample in self.convdown:
            x = downsample(x)
            skip_connections.append(x)
            x = nn.MaxPool2d(kernel_size=2)(x)

        x = self.bottleNeck(x)
        skip_connections.reverse()

        for idx in range(0, len(self.convup), 2):  ##2 step iteration
            x = self.convup[idx](x)
            skip_connection: object = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.convup[idx + 1](concat_skip)

        return self.convOut(x)


def main() -> object:
    """
    :rtype: None
    """
    model = Unet(in_channels=3, n_classes=3)
    print(model)
    tensor = torch.randn(1, 3, 128, 128)
    print(model(tensor).shape)


if __name__ == '__main__':
    main()
