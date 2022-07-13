import math
from typing import Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn.init import constant_, normal_
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, LambdaLR, _LRScheduler


def get_scheduler(optimizer: torch.optim.Optimizer,
                  n_iter_per_epoch: int,
                  epochs: int = 60,
                  lr_scheduler: str = 'step',
                  gamma: float = 0.1,
                  lr_steps: List[int] = [30, 45, 55],
                  warmup_epoch: int = 0,
                  warmup_multiplier: float = 100.0) -> _LRScheduler:
    """TDN schedulers

    Args:
        n_iter_per_epoch (int): `len(train_loader)`
    """
    scheduler: _LRScheduler
    if "cosine" in lr_scheduler:
        scheduler = CosineAnnealingLR(optimizer=optimizer,
                                      eta_min=0.00001,
                                      T_max=(epochs - warmup_epoch) * n_iter_per_epoch)
    elif "step" in lr_scheduler:
        scheduler = MultiStepLR(
            optimizer=optimizer,
            gamma=gamma,
            milestones=[(m - warmup_epoch) for m in lr_steps])
    else:
        raise NotImplementedError(f"scheduler {lr_scheduler} not supported")

    if warmup_epoch != 0:
        scheduler = GradualWarmupScheduler(optimizer,
                                           multiplier=warmup_multiplier,
                                           after_scheduler=scheduler,
                                           warmup_epoch=warmup_epoch * n_iter_per_epoch)

    return scheduler


def tsn_optim_policies(model: nn.Module) -> List[dict]:
    """Get Temporal Segment Network optimizer policies."""

    first_conv_weight = []
    first_conv_bias = []
    normal_weight = []
    normal_bias = []
    lr5_weight = []
    lr10_bias = []
    bn = []
    custom_ops: List[nn.Parameter] = []
    inorm: List[nn.Parameter] = []
    conv_cnt = 0
    bn_cnt = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or isinstance(
                m, torch.nn.Conv3d):
            ps = list(m.parameters())
            conv_cnt += 1
            if conv_cnt == 1:
                first_conv_weight.append(ps[0])
                if len(ps) == 2:
                    first_conv_bias.append(ps[1])
            else:
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
        elif isinstance(m, torch.nn.Linear):
            ps = list(m.parameters())
            if model.fc_lr5:
                lr5_weight.append(ps[0])
            else:
                normal_weight.append(ps[0])
            if len(ps) == 2:
                if model.fc_lr5:
                    lr10_bias.append(ps[1])
                else:
                    normal_bias.append(ps[1])

        elif isinstance(m, torch.nn.BatchNorm2d):
            bn_cnt += 1
            # later BN's are frozen
            if not model._enable_pbn or bn_cnt == 1:
                bn.extend(list(m.parameters()))
        elif isinstance(m, torch.nn.BatchNorm3d):
            bn_cnt += 1
            # later BN's are frozen
            if not model._enable_pbn or bn_cnt == 1:
                bn.extend(list(m.parameters()))
        elif len(m._modules) == 0:
            if len(list(m.parameters())) > 0:
                raise ValueError(
                    "New atomic module type: {}. Need to give it a learning policy".
                    format(type(m)))

    if model.fc_lr5:  # fine_tuning for UCF/HMDB
        return [
            {
                'params': first_conv_weight,
                'lr_mult': 1,
                'decay_mult': 1,
                'name': "first_conv_weight"
            },
            {
                'params': first_conv_bias,
                'lr_mult': 2,
                'decay_mult': 0,
                'name': "first_conv_bias"
            },
            {
                'params': normal_weight,
                'lr_mult': 1,
                'decay_mult': 1,
                'name': "normal_weight"
            },
            {
                'params': normal_bias,
                'lr_mult': 2,
                'decay_mult': 0,
                'name': "normal_bias"
            },
            {
                'params': bn,
                'lr_mult': 1,
                'decay_mult': 0,
                'name': "BN scale/shift"
            },
            {
                'params': custom_ops,
                'lr_mult': 1,
                'decay_mult': 1,
                'name': "custom_ops"
            },
            {
                'params': lr5_weight,
                'lr_mult': 5,
                'decay_mult': 1,
                'name': "lr5_weight"
            },
            {
                'params': lr10_bias,
                'lr_mult': 10,
                'decay_mult': 0,
                'name': "lr10_bias"
            },
        ]
    else:  # default
        return [
            {
                'params': first_conv_weight,
                'lr_mult': 1,
                'decay_mult': 1,
                'name': "first_conv_weight"
            },
            {
                'params': first_conv_bias,
                'lr_mult': 2,
                'decay_mult': 0,
                'name': "first_conv_bias"
            },
            {
                'params': normal_weight,
                'lr_mult': 1,
                'decay_mult': 1,
                'name': "normal_weight"
            },
            {
                'params': normal_bias,
                'lr_mult': 2,
                'decay_mult': 0,
                'name': "normal_bias"
            },
            {
                'params': bn,
                'lr_mult': 1,
                'decay_mult': 0,
                'name': "BN scale/shift"
            },
            {
                'params': custom_ops,
                'lr_mult': 1,
                'decay_mult': 1,
                'name': "custom_ops"
            },
        ]


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
      Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
      Args:
          optimizer (Optimizer): Wrapped optimizer.
          multiplier: init learning rate = base lr / multiplier
          warmup_epoch: target learning rate is reached at warmup_epoch, gradually
          after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
      """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 multiplier: float,
                 warmup_epoch: int,
                 after_scheduler: _LRScheduler,
                 last_epoch: int = -1):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.warmup_epoch = warmup_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch > self.warmup_epoch:
            return self.after_scheduler.get_lr()
        else:
            return [
                base_lr / self.multiplier *
                ((self.multiplier - 1.) * self.last_epoch / self.warmup_epoch + 1.)
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        if epoch > self.warmup_epoch:
            self.after_scheduler.step(epoch - self.warmup_epoch)
        else:
            super(GradualWarmupScheduler, self).step(epoch)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """

        state = {
            key: value
            for key, value in self.__dict__.items()
            if key != 'optimizer' and key != 'after_scheduler'
        }
        state['after_scheduler'] = self.after_scheduler.state_dict()
        return state

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """

        after_scheduler_state = state_dict.pop('after_scheduler')
        self.__dict__.update(state_dict)
        self.after_scheduler.load_state_dict(after_scheduler_state)
