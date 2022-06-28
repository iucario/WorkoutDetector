import torch
from fvcore.common.registry import Registry
import torchvision.transforms as T

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for dataset.

The registered object will be called with `obj(cfg, split)`.
The call should return a `torch.utils.data.Dataset` object.
"""

data_transforms = {
    'train':
        T.Compose([
            T.ToPILImage(),
            T.Resize(256),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    'test':
        T.Compose([
            T.ToPILImage(),
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
}


def build_dataset(dataset_name: str, cfg, split: str) -> torch.utils.data.Dataset:
    """
    Build a dataset, defined by `dataset_name`.
    Args:
        dataset_name (str): the name of the dataset to be constructed.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
    Returns:
        Dataset: a constructed dataset specified by dataset_name.
    """
    if dataset_name == 'FrameDataset':
        if split == 'train':
            anno_path = cfg.train.anno
            prefix = cfg.train.data_prefix
            transform = data_transforms['train']
        elif split == 'val':
            anno_path = cfg.val.anno
            prefix = cfg.val.data_prefix
            transform = data_transforms['test']
        elif split == 'test':
            anno_path = cfg.test.anno
            prefix = cfg.test.data_prefix
            transform = data_transforms['test']

        dataset = DATASET_REGISTRY.get('FrameDataset')(
            data_root=cfg.data_root,
            anno_path=anno_path,
            data_prefix=prefix,
            num_segments=cfg.num_segments,
            filename_tmpl=cfg.filename_tmpl,
            transform=transform,
            anno_col=cfg.anno_col,
        )
        return dataset
    elif dataset_name == 'ImageDataset':
        if split == 'train':
            anno_path = cfg.train.anno
            prefix = cfg.train.data_prefix
            transform = data_transforms['train']
        elif split == 'val':
            anno_path = cfg.val.anno
            prefix = cfg.val.data_prefix
            transform = data_transforms['test']
        elif split == 'test':
            anno_path = cfg.test.anno
            prefix = cfg.test.data_prefix
            transform = data_transforms['test']

        dataset = DATASET_REGISTRY.get('ImageDataset')(data_root=cfg.data_root,
                                                       data_prefix=prefix,
                                                       anno_path=anno_path,
                                                       transform=transform)
        return dataset
    return DATASET_REGISTRY.get(dataset_name)(cfg)
