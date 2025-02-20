import math
import torch

from torch import nn as nn



class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        """Basic block in U-net
        view: https://arxiv.org/abs/1505.04597 for detail

        Args:
            in_channels (int): num of input channels
            out_channels (int): num of output channels
        """
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
    
    
    
class Pos_embedder(nn.Module):
    
    def __init__(self, embedding_dim, temperature=2.7):
        """The positional embedding which expand the data dimension as map: R -> R^{embedding_dim}
        view https://arxiv.org/abs/1706.03762 for detail

        Args:
            embedding_dim (int): target dimension after embedding
            temperature (float, optional): temperature value. Defaults to 2.7.
        """
        super(Pos_embedder, self).__init__()
        self.embedding_dim = embedding_dim
        freq = torch.exp(-torch.arange(0, self.embedding_dim, 2)* math.log(temperature) / self.embedding_dim).view(-1, 1)
        self.register_buffer('freq', freq)

    def forward(self, pos_grid):
        
        h, w = pos_grid.size()
        if self.embedding_dim == 1:
            return pos_grid.reshape(1, h, w)
        
        pos_grid = pos_grid.reshape(1, -1)
        res = pos_grid.new_zeros((self.embedding_dim, h * w))
        
        res[0::2, :] = torch.sin(self.freq * pos_grid)
        res[1::2, :] = torch.cos(self.freq * pos_grid)
        res = res.view(self.embedding_dim, h, w)
        return res