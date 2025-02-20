from torch import nn as nn 

from .heads import build_head
from .backbones import build_backbone



class LowerLimb_Network(nn.Module):

    def __init__(self, args):
            super(LowerLimb_Network, self).__init__()
            self.args = args
            self.backbone = build_backbone(args.backbone)
            self.render_head, self.sdf_head = build_head(args.head)
        
    def forward(self, x):
        x = self.backbone(x)
        # {'coarse_pred': down_pred, 'fine_pred': out, 'coordi': points}
        preds = self.render_head(x)
        if self.sdf_head is not None:
            sdm = self.sdf_head(x, preds['coarse_pred'])
            preds['sdm'] = sdm
        return preds
    
    def freeze(self):
        if self.args.get('freeze_strategy', None) is not None:
            if self.args.freeze_strategy == 'encoder':
                for _, child in self.backbone.downs.named_children():
                    for param in child.parameters():
                        param.requires_grad = False
            else:
                for _, child in self.backbone.named_children():
                    for param in child.parameters():
                        param.requires_grad = False
                