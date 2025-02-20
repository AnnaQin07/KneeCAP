import torch 

from torch import nn as nn
from torch.nn import functional as F


class Render_head(nn.Module):
    
    def __init__(self, in_channels, num_classes=3, up_ratio=2, over_sampling_rate=3, ratio_importance=0.75, mode='train'):
        """The Render head which oversampling the features with high uncertainty scores
        view https://arxiv.org/abs/1912.08193 for detail
        Args:
            in_channels (int): the number of input channels, usually the number of feature maps
            num_classes (int, optional): number of classes. Defaults to 3.
            up_ratio (int, optional): determine the number of final sampling points. Defaults to 2.
            over_sampling_rate (int, optional): determine the number of inital sampling points. Defaults to 3.
            ratio_importance (float, optional): the top ratio_importance * init sample points points be choosen 
                                                according to their uncertainty scores. Defaults to 0.75.
            mode (str, optional): train or inf . Defaults to 'train'.
        """
        super(Render_head, self).__init__()
        self.up_ratio = up_ratio
        self.over_sampling_rate = over_sampling_rate
        self.ratio_importance = ratio_importance
        self.downhead = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.sample_head = nn.Conv1d(in_channels + num_classes, num_classes, kernel_size=1)
        self.mode = mode
        
    @torch.no_grad()
    def inference(self, x, down_pred):
        # up sample the down pred results
        num_points = 8096
        out = F.interpolate(down_pred, scale_factor=2, mode="bilinear", align_corners=True)
        points_idx, points = sampling_points(out, num_points, training=(self.mode=='train'))
        
        coarse_pred_feature = point_sample(out, points, align_corners=False)
        sampled_features = point_sample(x, points, align_corners=False)
        feature_representation = torch.cat([coarse_pred_feature, sampled_features], dim=1)
        rend = self.sample_head(feature_representation)

        B, C, H, W = out.shape
        points_idx = points_idx.unsqueeze(1).expand(-1, C, -1)
        out = (out.reshape(B, C, -1).scatter_(2, points_idx, rend).view(B, C, H, W))
        
        return {'coarse_pred': down_pred, 'fine_pred': out, 'coordi': points}

    
    def train_forward(self, down_pred, h, x):
        points = sampling_points(down_pred, h * self.up_ratio, self.over_sampling_rate, self.ratio_importance, training=(self.mode=='train'))

        # we combine the result of coarse predict with the feature-map features
        coarse_pred_feature = point_sample(down_pred, points, align_corners=False)
        sampled_features = point_sample(x, points, align_corners=False)
        feature_representation = torch.cat([coarse_pred_feature, sampled_features], dim=1)
        fine_pred = self.sample_head(feature_representation)
        return {'coarse_pred': down_pred, 'fine_pred': fine_pred, 'coordi': points}
         
    
    def forward(self, x):
        
        down_pred =  self.downhead(x)
        b, c, h, w = down_pred.shape
        if self.mode == 'train':
            return self.train_forward(down_pred, h, x)
        else:
            return self.inference(x, down_pred)
            
            

def point_sample(inputs, point_coords, **kwargs):
    """Pick up the features of inputted pixel coordinates

    Args:
        inputs (torch.tensor([B, C, H, W])): the feature map or score map
        point_coords (torch.tensor([B, num_samplers, 2])): pixel coordinates

    Returns:
        torch.tensor([B, num_samplers, C]): picked features /scores
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(inputs, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output




@torch.no_grad()
def sampling_points(mask, n_sample, k=3, beta=0.75, training=True):
    """Point sampling operations

    Args:
        mask (torch.tensor([B, C, H, W])): the coarse predicted masks, C presents the number of classes 
        n_sample (int): number of output points
        k (int, optional): inital oversampling rate. Defaults to 3.
        beta (float, optional): select ratio of topK uncertain points within the inital oversampling points. Defaults to 0.75.
        training (bool, optional): whether it's in training mode. Defaults to True.

    Returns:
        torch.tensor([B, n_sample 2]): the pixel coordinates of choosing sampler points
    """
    assert mask.dim() == 4, "dim must be BCHW"
    device = mask.device 
    B, _, H, W = mask.shape 
    mask, _ = mask.sort(dim=1, descending=True)
    
    if not training:
        # we don't oversampling during inference
        H_step, W_step = 1 / H, 1 / W
        n_sample = min(H * W, n_sample)
        uncertaintly_map = -1 * (mask[:, 0] - mask[:, 1])        
        _, idx = uncertaintly_map.view(B, -1).topk(n_sample, dim=1)
        
        points = torch.zeros(B, n_sample, 2, dtype=torch.float32, device=device)
        points[:, :, 0] = W_step / 2.0 + (idx  % W).to(torch.float) * W_step
        points[:, :, 1] = H_step / 2.0 + torch.div(idx, W, rounding_mode='trunc').to(torch.float) * H_step
        return idx, points 
    # initally choosing k * n_sample points and select beta * n_sample points with higher uncertainty scores 
    over_generation = torch.rand((B, k * n_sample, 2), device=device)
    over_generation_map = point_sample(mask, over_generation, align_corners=False)
    uncertaintly_map = -1 * (over_generation_map[:, 0] - over_generation_map[:, 1])
    _, idx = uncertaintly_map.topk(int(beta * n_sample), -1)
    shift = (k * n_sample) * torch.arange(B, dtype=torch.long, device=device)
    idx += shift[:, None]
    importance = over_generation.view(-1, 2)[idx.view(-1), :].view(B, int(beta * n_sample), 2)
    
    # randomly choosing (1 - beta) * n_sample points as negative samplers
    coverage = torch.rand(B, n_sample - int(beta * n_sample), 2, device=device)
    return torch.cat([importance, coverage], 1).to(device)