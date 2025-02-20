import torch
from torch import nn as nn 

import torchvision.transforms.functional as TF

from .basic_blocks import DoubleConv

class Asymmetric_UNet(nn.Module):
    def __init__(self, in_channels=1, 
                 in_features=[32, 64, 128, 256, 512],
                 out_features=[512, 256, 128, 64]):
        """The Asymmetric U-net whose number of encoding block is greater than the decoding block

        Args:
            in_channels (int, optional): number of channels of input images. Defaults to 1.
            in_features (list, optional): number of output channels of encoding blocks. Defaults to [32, 64, 128, 256, 512].
            out_features (list, optional): number of output channels of decoding blocks. Defaults to [512, 256, 128, 64].
        """
        super(Asymmetric_UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part (encoding) of U-NET
        for feature in in_features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part (decoding) of U-NET
        for feature in out_features:
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(in_features[-1], in_features[-1] * 2)

    def forward(self, x, encodes_only=False):
        skip_connections = []

        for i, down in enumerate(self.downs):
            x = down(x)
            if i > 0:
                skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        if encodes_only:
            return x
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            # skip connection
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)
        return x