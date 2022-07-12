from typing import Iterable, List, Tuple, Union
import torch
import torch.nn as nn
from fvcore.common.config import CfgNode
from fvcore.common.registry import Registry
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import StepLR, _LRScheduler

from .optimizer import get_scheduler, tsn_optim_policies

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """
Registry for model.

The registered object will be called with `obj(cfg)`.
The call should return a `torch.nn.Module` object.
"""


def build_model(cfg) -> torch.nn.Module:
    """Build a model, defined by `model_name`.
    
    Args:
        cfg (CfgNode): configs. Details can be found in configs/defaults.yaml

    Returns:
        torch.nn.Module: the model.
    """

    if cfg.model.model_type.lower() == 'tdn':
        from .tdn import create_model as create_model_tdn
        model = create_model_tdn(**cfg.model)
    elif cfg.model.model_type.lower() == 'tsm':
        from .tsm import create_model as create_model_tsm
        model = create_model_tsm(**cfg.model)
    else:
        raise KeyError(f"Model '{cfg.model.model_type}' is not supported.")
    return model


def build_optim(cfg: CfgNode, model: nn.Module,
                **kwargs) -> Tuple[torch.optim.Optimizer, _LRScheduler]:
    """Build optimizer, scheduler, and warmup_scheduler.
    
    Args:
        cfg (CfgNode): configs. Details can be found in configs/defaults.yaml
        model (nn.Module): the model.
        kwargs (dict):
            - n_iter_per_epoch (int): `len(train_loader)`. For TDN.
    Returns:
        tuple: optimizer, scheduler, and warmup_scheduler.
    Note:
        TDN has its own optimizer and scheduler.
        TSM is the same as TSN, has its own optimize policy.
    """
    OPTIMIZER = cfg.optimizer.method.lower()
    POLICY = cfg.lr_scheduler.policy.lower()
    optimizer: torch.optim.Optimizer
    scheduler: _LRScheduler
    params: Union[list, Iterable]

    if OPTIMIZER == 'tsn':  # policy for TSM and TSN
        params = tsn_optim_policies(model)
        optimizer = SGD(params,
                        lr=cfg.optimizer.lr,
                        momentum=cfg.optimizer.momentum,
                        weight_decay=cfg.optimizer.weight_decay)
    elif OPTIMIZER == 'sgd':
        optimizer = SGD(model.parameters(),
                        lr=cfg.optimizer.lr,
                        momentum=cfg.optimizer.momentum,
                        weight_decay=cfg.optimizer.weight_decay)
    elif OPTIMIZER == 'adamw':
        optimizer = AdamW(model.parameters(),
                          lr=cfg.optimizer.lr,
                          eps=cfg.optimizer.eps,
                          weight_decay=cfg.optimizer.weight_decay)
    else:
        raise NotImplementedError(f'Not implemented optimizer: {cfg.optimizer.method}',
                                  f'Supported optimizers: {["tsn", "sgd", "adamw"]}')

    if POLICY == 'tdn':
        assert 'n_iter_per_epoch' in kwargs, \
            "n_iter_per_epoch is required for TDN scheduler."
        scheduler = get_scheduler(optimizer,
                                  kwargs['n_iter_per_epoch'],
                                  epochs=cfg.trainer.max_epochs,
                                  lr_steps=cfg.lr_scheduler.steps,
                                  gamma=cfg.lr_scheduler.gamma,
                                  warmup_epoch=cfg.lr_scheduler.warmup_epoch)
    elif POLICY == 'steplr':
        scheduler = StepLR(optimizer,
                           step_size=cfg.lr_scheduler.step,
                           gamma=cfg.lr_scheduler.gamma)
    else:
        raise NotImplementedError(f'Not implemented lr schedular: {cfg.lr_schedular}',
                                  f'Supported lr schedulers: {["tdn", "steplr"]}')
    return optimizer, scheduler