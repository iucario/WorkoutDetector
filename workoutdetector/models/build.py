import torch
from fvcore.common.registry import Registry

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
