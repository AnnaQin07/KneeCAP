import math 
import torch 

from torch import nn as nn 
from torch.nn import init as init



class SDF_head(nn.Module):
    
    def __init__(self, num_featues, num_classes, pos_embd_channel=1, pos_embd_temp=2.7, ratio=[1024, 128], mlp_channels=[32, 16]):
        """SDF head which predict Signed Distance Function map

        Args:
            num_featues (int): feature map channels
            num_classes (int): number of classes
            pos_embd_channel (int, optional): channels after positional embedding. Defaults to 1.
            pos_embd_temp (float, optional): temperture of embedding. Defaults to 2.7.
            ratio (list, optional): [height width] ratio of feature maps. Defaults to [1024, 128].
            mlp_channels (list, optional): output channels of mlp. Defaults to [32, 16].
        """
        super(SDF_head, self).__init__()
        self.num_classes = num_classes
        self.ratio = ratio
        self.pos_embder = Pos_embedder(pos_embd_channel, pos_embd_temp)
        self.edge_layer = Edge_encoder(num_classes-1)
        # self.channel_conv_layers = nn.Sequential(*build_channel_convs(in_channels + 2*num_classes + 2*pos_embd_channel - 1, mlp_channels))
        self.sample_head = nn.Conv1d(num_featues+num_classes, num_classes, kernel_size=1)
        self.channel_mlp = Channel_MLP(num_featues+2*num_classes+2*pos_embd_channel-1, mlp_channels)
        self.sdf_layer = nn.Conv2d(mlp_channels[-1], num_classes-1, kernel_size=1)
        self.generate_grid()
    
    def generate_grid(self):
        heights = torch.arange(self.ratio[0])
        widths = torch.arange(self.ratio[1])
        grid_h, grid_w = torch.meshgrid(heights, widths, indexing='ij')

        grid_h = self.pos_embder(grid_h)
        grid_w = self.pos_embder(grid_w)
        pos_grid = torch.cat([grid_h, grid_w], dim=0)
        self.register_buffer('pos_grid', pos_grid)
        

    def forward(self, x, seg_out):
        """forward the seg-sdf head

        Args:
            x (torch.tensor(shape=[b, c, h, w])): featuremap from vision backbone

        Returns:
            torch.tensor(shape=[b, 3, h, w]): segmentation mask, channel: [tibial, femur, background]
            torch.tensor(shape=[b, 2, h, w]): sdf mask, channel: [tibial, femur]
        """
        bsz = x.size(0)
        edge_embedding = self.edge_layer(seg_out[:, :2])
        sdf_in = torch.cat([x, seg_out, edge_embedding, self.pos_grid.repeat(bsz, 1, 1, 1)], dim=1)
        sdf_feature = self.channel_mlp(sdf_in)
        return self.sdf_layer(sdf_feature)



class Edge_encoder(nn.Module):
    def __init__(self, num_channels):
        """Edge encoder as the bottle-neck of segmentation masks
        structure: num_channels -> 4*num_channels -> num_channels
        Args:
            num_channels (int): number of input plus output channels
        """
        super(Edge_encoder, self).__init__()
        self.num_channels = num_channels 
        self.layer = nn.Sequential(nn.Conv2d(num_channels, num_channels*4, 3, 1, 1, groups=2, bias=False),
                                   nn.BatchNorm2d(num_channels*4),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(num_channels*4, num_channels, 3, 1, 1, groups=2, bias=False),
                                   nn.BatchNorm2d(num_channels),
                                   nn.ReLU(inplace=True))
    def forward(self, x):
        return self.layer(x)
    
class Pos_embedder(nn.Module):
    
    def __init__(self, embedding_dim, temperature=2.7):
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


    
class Channel_MLP(nn.Module):
    def __init__(self, in_channel, mlp_channels, have_bn=True):
        """MLP for SDF map prediction

        Args:
            in_channel (int): number of input channels 
            mlp_channels (list): a list of mlp output channels
            have_bn (bool, optional): whether has batch norm layer. Defaults to True.
        """
        super(Channel_MLP, self).__init__()
        self.in_channel = in_channel
        layers = []
        for channel in mlp_channels:
            conv = nn.Conv2d(in_channel, channel, kernel_size=1)
            # init.xavier_normal_(conv.weight, gain=1.0)
            in_channel = channel
            layers.append(conv)
            if have_bn:
                layers.append(nn.BatchNorm2d(channel))
            layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)
        self.customize_init()
    
    def customize_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight, gain=1.0)
    
    def forward(self, x):
        return self.layers(x)
