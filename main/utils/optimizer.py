import math
import torch

from bisect import bisect_right
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR


class _LRScheduler(object):
    def __init__(self, optimizer, base_lr, last_iter=-1):
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(f'{type(optimizer).__name__} is not an Optimizer')
        self.optimizer = optimizer
        self.base_lrs = [base_lr for _ in optimizer.param_groups]
        self.last_iter = last_iter

    def _get_new_lr(self):
        raise NotImplementedError

    def get_lr(self):
        return list(map(lambda group: group['lr'], self.optimizer.param_groups))

    def step(self, this_iter=None):
        if this_iter is None:
            this_iter = self.last_iter + 1
        self.last_iter = this_iter
        for param_group, lr in zip(self.optimizer.param_groups, self._get_new_lr()):
            param_group['lr'] = lr


class CLRScheduler(_LRScheduler):
    '''
    Cyclical Learning Rates for Training Neural Networks
    view the paper: https://arxiv.org/abs/1506.01186 for more details
    '''
    def __init__(self, optimizer, base_lr, max_lr, step_size, scale_mode='const', gamma=1., last_iter=-1):
        self.scale_mode = scale_mode
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.gamma = gamma
        super(CLRScheduler, self).__init__(optimizer, base_lr, last_iter)

    def _get_new_lr(self):
        cycle = math.floor(1 + (self.last_iter - 1) / (2 * self.step_size))
        x = abs((self.last_iter - 1) / self.step_size - 2 * cycle + 1)
        return [base_lr + (self.max_lr - base_lr) * max(0, (1 - x)) * self.scale_fn(x) for base_lr in self.base_lrs]

    def scale_fn(self, x):
        if self.scale_mode == 'const':
            return x * 4.
        elif self.scale_mode == 'linear_decrease':
            return x * 4 / 2**((self.last_iter - 1) // self.step_size)
        elif self.scale_mode == 'exp':
            scale = (self.last_iter - 1) // self.step_size
            return (x * 4) / ((1 + scale * self.gamma) ** scale)
        else:
            raise ValueError



class _WarmUpLRScheduler(_LRScheduler):

    def __init__(self, optimizer, base_lr, warmup_lr, warmup_steps, last_iter=-1):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        if warmup_steps == 0:
            self.warmup_lr = base_lr
        else:
            self.warmup_lr = warmup_lr
        super(_WarmUpLRScheduler, self).__init__(optimizer, base_lr, last_iter)

    def _get_warmup_lr(self):
        if self.warmup_steps <= 0 or self.last_iter >= self.warmup_steps:
             return None
        # first compute relative scale for self.base_lr, then multiply to base_lr
        scale = ((self.last_iter / self.warmup_steps) * (
                    self.warmup_lr - self.base_lr) + self.base_lr) / self.base_lr
        # print('last_iter: {}, warmup_lr: {}, base_lr: {}, scale: {}'.format(self.last_iter, self.warmup_lr, self.base_lr, scale))
        return [scale * base_lr for base_lr in self.base_lrs]


class StepLRScheduler(_WarmUpLRScheduler):
    def __init__(self, optimizer, milestones, lr_mults, base_lr, warmup_lr, warmup_steps, last_iter=-1):
        super(StepLRScheduler, self).__init__(optimizer, base_lr, warmup_lr, warmup_steps, last_iter)

        assert len(milestones) == len(lr_mults), f"{milestones} vs {lr_mults}"
        for x in milestones:
            assert isinstance(x, int)
        if list(milestones) != sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)
        self.milestones = milestones
        self.lr_mults = [1.0]
        self.lr_mults.extend(self.lr_mults[-1] * x for x in lr_mults)

    def _get_new_lr(self):
        warmup_lr = self._get_warmup_lr()
        if warmup_lr is not None:
            return warmup_lr

        pos = bisect_right(self.milestones, self.last_iter)
        scale = self.warmup_lr * self.lr_mults[pos] / self.base_lr
        return [base_lr * scale for base_lr in self.base_lrs]


class CosineLRScheduler(_WarmUpLRScheduler):
    def __init__(self, optimizer, T_max, eta_min, base_lr, warmup_lr, warmup_steps, last_iter=-1):
        super(CosineLRScheduler, self).__init__(optimizer, base_lr, warmup_lr, warmup_steps, last_iter)
        self.T_max = T_max
        self.eta_min = eta_min

    def _get_new_lr(self):
        warmup_lr = self._get_warmup_lr()
        if warmup_lr is not None:
            return warmup_lr

        step_ratio = (self.last_iter - self.warmup_steps) / (self.T_max - self.warmup_steps)
        target_lr = self.eta_min + (self.warmup_lr - self.eta_min) * (1 + math.cos(math.pi * step_ratio)) / 2
        scale = target_lr / self.base_lr
        return [scale * base_lr for base_lr in self.base_lrs]




def build_optimizer(model, args):
    if args.name == 'Adam':
        return Adam(model.parameters(), lr=args.lr, weight_decay=getattr(args, 'weight_dacay', 0))
    elif args.name == 'sgd':
        return SGD(model.parameters(), 
                   lr=args.lr, 
                   momentum=getattr(args, 'momentum', 0),
                   weight_decay=getattr(args, 'weight_dacay', 0))
    else:
        raise ValueError("unrecognised optimizer type")


def build_scheduler(optimizer, args):
    if args.name == "ReduceLROnPlateau":
        return ReduceLROnPlateau(optimizer, 
                                 args.mode, 
                                 factor=args.factor, patience=args.patience)
    elif args.name == 'step':
        return StepLRScheduler(optimizer,
                               args.milestones, 
                               args.lr_mults, 
                               args.base_lr, 
                               args.warmup_lr, 
                               args.warmup_steps, 
                               last_iter=getattr(args, 'last_iter', -1))
    elif args.name == 'cosine':
        return CosineAnnealingLR(optimizer,
                                 args.T_max, 
                                 args.eta_min, 
                                 args.base_lr, 
                                 args.warmup_lr, 
                                 args.warmup_steps, 
                                 last_iter=getattr(args, 'last_iter', -1))
    else:
        raise ValueError("unrecognised lr scheduler type")
        
    

# optimizer = optim.Adam(model.parameters(), lr=args.lr)
# scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=5)