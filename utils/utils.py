import os
from bisect import bisect_right

import torch
import torch.optim as optim
from typing import List



def get_optimizer(cfg, parameters):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            [{'params': parameters, 'initial_lr': cfg.TRAIN.LR}],
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            parameters,
            lr=cfg.TRAIN.LR,
#             weight_decay=cfg.TRAIN.WD,
        )
    elif cfg.TRAIN.OPTIMIZER == 'adamw':
        optimizer = optim.AdamW(
            parameters,
            lr=cfg.TRAIN.LR,
            weight_decay=cfg.TRAIN.WD,
        )

    return optimizer


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth.tar'):

    torch.save(states, os.path.join(output_dir, 'model', filename))

    if is_best and 'state_dict' in states:
        torch.save(
            states['best_state_dict'],
            os.path.join(output_dir, 'src','model_best.pth.tar')
        )



class LinearWarmupStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        milestones: List[int],
        gamma: float = 0.1,
        warmup_ratio: float = 0.1,
        warmup_iters: int = 1000,
        warmup_method: str = 'linear',
        last_epoch: int = -1,
        cur_iters: int = -1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}", milestones
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_ratio = warmup_ratio
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.cur_iters = cur_iters
        super().__init__(optimizer, last_epoch)
        self.step_iter()

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.cur_iters, self.warmup_iters, self.warmup_ratio
        )
        return [
            base_lr * warmup_factor * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs

        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()

    def step_iter(self):

        self.cur_iters += 1
        if self.cur_iters < self.warmup_iters:
            values = self._compute_values()
            for i, data in enumerate(zip(self.optimizer.param_groups, values)):
                param_group, lr = data
                param_group['lr'] = lr

            self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    # def _init_lr(self):
    #     for i, data in enumerate(self.optimizer.param_groups):
    #         param_group = data
    #         param_group['lr'] = 0.00025
    #
    #     self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

def _get_warmup_factor_at_iter(method: str, iters: int, warmup_iters: int, warmup_ratio: float) -> float:
    """
    Return the learning rate warmup factor at a specific iteration.
    See https://arxiv.org/abs/1706.02677 for more details.

    Args:
        method (str): warmup method; either "constant" or "linear".
        iters (int): iteration at which to calculate the warmup factor.
        warmup_iters (int): the number of warmup iterations.
        warmup_ratio (float): the base warmup factor (the meaning changes according
            to the method used).

    Returns:
        float: the effective warmup factor at the given iteration.
    """
    if iters >= warmup_iters:
        return 1.0

    if method == "constant":
        return warmup_ratio
    elif method == "linear":
        return 1 - (1 - warmup_ratio) * (1 - iters / warmup_iters)
    elif method == 'exp':
        return warmup_ratio**(1 - iters / warmup_iters)
    else:
        raise ValueError("Unknown warmup method: {}".format(method))
