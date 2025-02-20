import torch
import torch.nn as nn


class Losses(object):
    
    def __init__(self, args):
        """Create a list of losses

        Args:
            args (EasyDict): loss configs
        """
        losses = {}
        for i, item in enumerate(args.items):
            if item == 'ce':
                losses[item] = (args.weights[i], 
                            nn.CrossEntropyLoss())
            elif item == 'focal':
                losses[item] = (args.weights[i],
                                nn.FocalLoss(args.class_num, alpha=args.get('alpha', None), gamma=args.get('gamma', 2), size_average=args.get('size_average', True)))
            elif item == 'dice':
                losses[item] = (args.weights[i], 
                            DiceLoss(args.smooth, args.get('ignore_channel', None)))
            elif item == 'mse':
                losses[item] = (args.weights[i], 
                            nn.MSELoss(reduction='mean'))

        self.losses = losses
    
    def __call__(self, inputs, targets, loss_func):

        return self.losses[loss_func][0] * self.losses[loss_func][1](inputs, targets)

            
            
class DiceLoss(nn.Module):
    
    def __init__(self, smooth, ignore_channel):
        """Dice loss implementation
        view: https://arxiv.org/abs/1707.03237v3 for detail

        Args:
            smooth (float): smooth val in case of the zero value in denominator
            ignore_channel (int): ignore channel for loss computation
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_channel = ignore_channel
        
    def forward(self, inputs, targets):

        return 1 - self.dice_coefficient(inputs, targets)
    
    def dice_coefficient(self, inputs, targets):
        inputs = torch.softmax(inputs, dim=1)
        total_score = torch.tensor(0.0).to(inputs.device)
        c = inputs.shape[1]
        
        for i in range(c):
            if i != self.ignore_channel:
                intersec = torch.sum(inputs[:, i, :, :] * targets[:, i, :, :])
                addition = torch.sum(inputs[:, i, :, :]) + torch.sum(targets[:, i, :, :])
                score = (intersec * 2 + self.smooth) / (addition + self.smooth)
                total_score += score
        c = c- 1 if self.ignore_channel is not None else c
        return total_score / c