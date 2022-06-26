import torch
from fvcore.common.registry import Registry

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """
Registry for model.

The registered object will be called with `obj(cfg)`.
The call should return a `torch.nn.Module` object.
"""


def build_dataset(model_name: str, cfg) -> torch.nn.Module:
    """Build a model, defined by `model_name`.
    
    Args:
        model_name (str): the name of the model to be constructed.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py

    Returns:
        torch.nn.Module: the model.
    """

    return MODEL_REGISTRY.get(model_name)(cfg)
