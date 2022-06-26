import argparse
import math
import os
import os.path as osp
from os.path import join as osj
from typing import Callable, List, Optional, Tuple
import einops
import numpy as np
import timm
import torch
import torchvision.transforms as T
from torch import Tensor, nn, optim, utils
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.io import read_image
from workoutdetector.settings import PROJ_ROOT

from workoutdetector.datasets.build import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class FrameDataset(torch.utils.data.Dataset):
    """Frame dataset for video. 
    label.txt has the following format:
        `dir/to/video/frames start_index total_frames label`
    
    Labels are built using `scripts/build_label_list.py`

    Args:
        data_root (str): root directory of the dataset
        data_prefix (str): prefix relative to `data_root`. 
        The final data path is `data_root/data_prefix/frame_dir`
        num_segments (int): number of segments to sample from each video
        filename_tmpl (str): template of the frame filename, e.g. `img_{:05}.jpg`
        transform (callable, optional): transform to apply to the frames

    Returns:
        Tensor, shape (N, C, H, W)
        List[int], label
    """

    def __init__(self,
                 data_root: str,
                 anno_path: str,
                 data_prefix: Optional[str] = None,
                 num_segments: int = 8,
                 filename_tmpl='img_{:05}.jpg',
                 transform: Optional[Callable] = None) -> None:
        super().__init__()
        assert osp.isfile(anno_path), f'{anno_path} is not file'
        self.data_root = data_root
        self.data_prefix = osj(data_root, data_prefix if data_prefix else '')
        self.transform = transform
        self.num_segments = num_segments
        self.tmpl = filename_tmpl
        self.anno: List[dict] = self.load_annotation(anno_path)

    def load_annotation(self, anno_path: str) -> List[dict]:
        video_infos = []
        with open(anno_path, 'r') as f:
            for line in f:
                frame_dir, start_index, total_frames, label = line.split()
                if self.data_prefix is not None and int(total_frames) > 0:
                    frame_dir = os.path.join(self.data_prefix, frame_dir)
                video_infos.append(
                    dict(frame_dir=frame_dir,
                         start_index=int(start_index)+1,
                         total_frames=int(total_frames),
                         label=int(label)))
        return video_infos

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        frame_list = []
        video_info = self.anno[idx]
        frame_dir = video_info['frame_dir']
        start_index = video_info['start_index']
        total_frames = video_info['total_frames']
        label = video_info['label']
        samples = sample_frames(total_frames, self.num_segments, start_index)
        for i in samples:
            frame_path = os.path.join(frame_dir, self.tmpl.format(i))
            frame = read_image(frame_path)
            frame_list.append(frame)
        if self.transform is not None:
            frame_list = [self.transform(frame) for frame in frame_list]
        frame_tensor = torch.stack(frame_list, 0)
        assert frame_tensor.shape[0] == self.num_segments, \
            f'frame_list.shape[0] = {frame_tensor.shape[0]}, ' \
            f'but self.num_segments = {self.num_segments}'
        return frame_tensor, label

    def __len__(self) -> int:
        return len(self.anno)


@DATASET_REGISTRY.register()
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


def sample_frames(total: int, num: int, offset: int = 0) -> List[int]:
    """Uniformly sample num frames from video
    
    Args:
        total: int, total frames, 
        num: int, number of frames to sample
        offset: int, offset from start of video
    Returns: 
        list of frame indices starting from offset
    """

    if total < num:
        # repeat frames if total < num
        repeats = math.ceil(num / total)
        data = [x for x in range(total) for _ in range(repeats)]
        total = len(data)
    else:
        data = list(range(total))
    interval = total // num
    indices = np.arange(0, total, interval)[:num]
    for i, x in enumerate(indices):
        rand = np.random.randint(0, interval)
        if i == num - 1:
            upper = total
            rand = np.random.randint(0, upper - x)
        else:
            upper = min(interval * (i + 1), total)
        indices[i] = (x + rand) % upper
    assert len(indices) == num, f'len(indices)={len(indices)}'
    for i in range(1, len(indices)):
        assert indices[i] > indices[i - 1], f'indices[{i}]={indices[i]}'
    assert num == len(indices), f'num={num}'
    return [data[i] + offset for i in indices]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    data_root = os.path.join(PROJ_ROOT, 'data/RepCount/rawframes')
    anno_path = os.path.join(PROJ_ROOT, 'data/relabeled/pull_up/train.txt')
    dataset = FrameDataset(data_root,
                           anno_path=anno_path,
                           data_prefix=None,
                           num_segments=8)
    print(len(dataset))
    random_index = np.random.randint(0, len(dataset))
    img, label = dataset[random_index]
    plt.figure(figsize=(8, 4), dpi=200)
    img = einops.rearrange(img, '(b1 b2) c h w -> (b1 h) (b2 w) c', b2=4)
    plt.title(f'label: {label}')
    print(img.shape)
    plt.imshow(img)
    plt.show()