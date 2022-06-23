import argparse
import os
import os.path as osp
from os.path import join as osj
from typing import Callable, List, Optional, Tuple

import timm
import torch
import torchvision.transforms as T
from torch import Tensor, nn, optim, utils
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.io import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from workoutdetector.datasets import RepcountImageDataset
from workoutdetector.settings import PROJ_ROOT


class ImageDataset(torch.utils.data.Dataset):
    """General image dataset
    label text files of [image.png class] are required

    Args:
        data_root: str
        data_prefix: str, will be appended to data_root
        anno_path: str, abusolute annotation path
        transform: Optional[Callable], data transform
    """

    def __init__(self,
                 data_root: str,
                 data_prefix: str,
                 anno_path: str = 'train.txt',
                 transform: Optional[Callable] = None) -> None:
        super().__init__()
        assert osp.isfile(anno_path), f'{anno_path} is not file'
        self.data_root = data_root
        self.data_prefix = osj(data_root, data_prefix)
        self.transform = transform
        self.anno: List[Tuple[str, int]] = self.read_txt(anno_path)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        path, label = self.anno[index]
        img = read_image(osj(self.data_prefix, path))
        if self.transform:
            img = self.transform(img)
        return img, int(label)

    def __len__(self) -> int:
        return len(self.anno)

    def read_txt(self, path: str) -> List[Tuple[str, int]]:
        """Read annotation file 
        
        Args:
            path: str, path to annotation file

        Returns:
            List of [path, class]
        """
        ret = []
        with open(path) as f:
            for line in f:
                name, class_ = line.strip().split()
                ret.append((name, int(class_)))
        return ret
