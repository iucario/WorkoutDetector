from typing import Optional, Tuple
import torch
import torchvision.transforms as T
from fvcore.common.config import CfgNode
from .common import ImageDataset, FrameDataset
from .transform import PersonCrop, MultiScaleCrop


def build_dataset(cfg: CfgNode, split: str) -> torch.utils.data.Dataset:
    """
    Build a dataset, defined by `dataset_name`.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
    Returns:
        Dataset: a constructed dataset specified by dataset_name.
    """
    if cfg.dataset_type == 'FrameDataset':
        anno_path = cfg.get(split).anno
        prefix = cfg.get(split).data_prefix
        transform = cfg.get(split).transform

        return FrameDataset(
            data_root=cfg.data_root,
            anno_path=anno_path,
            data_prefix=prefix,
            num_segments=cfg.num_segments,
            filename_tmpl=cfg.filename_tmpl,
            transform=build_transform(split, person_crop=transform.person_crop),
            anno_col=cfg.anno_col,
        )
    elif cfg.dataset_type == 'ImageDataset':
        anno_path = cfg.get(split).anno
        prefix = cfg.get(split).data_prefix
        transform = cfg.get(split).transform

        return ImageDataset(
            data_root=cfg.data_root,
            data_prefix=prefix,
            anno_path=anno_path,
            transform=build_transform(split, person_crop=transform.person_crop),
        )

    else:
        raise KeyError(f"Dataset '{cfg.dataset_type}' is not supported.")


MEAN_STD = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
MULTI_SCALES = (1.0, 0.875, 0.75, 0.66)
INPUT_SIZE = (224, 224)


def build_transform(split: str, person_crop: bool = False) -> T.Compose:
    """Build transform for videos and images
    
    Args:
        dataset_type (str): 'FrameDataset' or 'ImageDataset'
        split (str): split name. If train, 
        use multi_scale_crop. If test, use person_crop.
        person_crop (bool): whether to use PersonCrop. If False, use center crop.
    Returns:
        T.Compose, transform
    """
    if split == 'train':
        return build_train_transform()
    else:
        return build_test_transform(person_crop=person_crop)


def build_train_transform(
        multi_scale_crop: Optional[Tuple[float, ...]] = MULTI_SCALES) -> T.Compose:
    """Build train transform
    
    Args:
        multi_scale_crop (None or List): list of scale sizes to crop
    Returns:
        T.Compose, train transform
    """
    if multi_scale_crop is None:
        return T.Compose([
            T.ConvertImageDtype(torch.float32),
            T.Resize(256),
            T.RandomCrop(INPUT_SIZE),
            T.RandomHorizontalFlip(),
            T.Normalize(**MEAN_STD),
        ])
    else:
        return T.Compose([
            T.ConvertImageDtype(torch.float32),
            MultiScaleCrop(scales=multi_scale_crop),
            T.Resize(INPUT_SIZE),
            T.RandomHorizontalFlip(),
            T.Normalize(**MEAN_STD),
        ])


def build_test_transform(person_crop: bool) -> T.Compose:
    """Build test transform
    
    Args:
        person_crop (bool): whether to use PersonCrop. If not use centor crop.
    Returns:
        T.Compose, test transform
    """
    if person_crop:
        return T.Compose([
            PersonCrop(),
            T.ConvertImageDtype(torch.float32),
            T.Resize(256),
            T.Normalize(**MEAN_STD),
        ])
    else:
        return T.Compose([
            T.ConvertImageDtype(torch.float32),
            T.Resize(256),
            T.CenterCrop(INPUT_SIZE),
            T.Normalize(**MEAN_STD),
        ])
