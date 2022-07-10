import os
import os.path as osp
from os.path import join as osj
from numpy.random import randint
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torchvision.io import read_image, read_video


class TDNDataset(torch.utils.data.Dataset):
    """TDN dataset. Samples frame idx and the frames around it.
    
    label.txt has the following format:
        `dir/to/video/frames start_index total_frames label`
    
    Labels are built using `scripts/build_label_list.py`

    Note:
        `start_index` should be 1-based.
        `anno_path` should be absolute path.

    Args:
        data_root (str): root directory of the dataset
        data_prefix (str): prefix relative to `data_root`. 
        The final data path is `data_root/data_prefix/frame_dir`
        num_segments (int): number of segments to sample from each video
        num_frames (int): number of frames around the center frame, default 5.
        filename_tmpl (str): template of the frame filename, e.g. `img_{:05}.jpg`
        transform (callable, optional): transform to apply to the frames
        anno_col (int): columns in the annotation file, default is 4
            3-column annotation file:
                `frame_dir total_frames label`
            4-column annotation file:
                `frame_dir start_index total_frames label`

    Returns:
        Tensor, shape (N*T, C, H, W)
        List[int], label
    
    Example:
    >>> data_root = '/data/RepCount/rawframes'
    >>> anno_path = '/data/relabeled/pull_up/train.txt'
    >>> dataset = TDNDataset(data_root, anno_path=anno_path,
    ...                      data_prefix=None, num_segments=8)
    >>> data, label = dataset[0]
    >>> data.shape
    torch.Size([40, 3, 256, 256])
    """

    def __init__(self,
                 data_root: str,
                 anno_path: str,
                 data_prefix: Optional[str] = None,
                 num_segments: int = 8,
                 num_frames: int = 5,
                 filename_tmpl='img_{:05}.jpg',
                 transform: Optional[Callable] = None,
                 anno_col: int = 4) -> None:
        super().__init__()
        assert osp.isfile(anno_path), f'{anno_path} is not file'
        self.data_root = data_root
        self.data_prefix = osj(data_root, data_prefix if data_prefix else '')
        self.transform = transform
        self.num_segments = num_segments
        self.num_frames = num_frames
        self.tmpl = filename_tmpl
        self.anno_col = anno_col
        self.anno: List[dict] = self.load_annotation(anno_path)

    def load_annotation(self, anno_path: str) -> List[dict]:
        video_infos = []
        with open(anno_path, 'r') as f:
            if self.anno_col == 4:
                for line in f:
                    frame_dir, start_index, total_frames, label = line.split()
                    if self.data_prefix is not None and int(total_frames) > 0:
                        frame_dir = os.path.join(self.data_prefix, frame_dir)
                    video_infos.append(
                        dict(frame_dir=frame_dir,
                             start_index=int(start_index),
                             total_frames=int(total_frames),
                             label=int(label)))
            elif self.anno_col == 3:
                for line in f:
                    frame_dir, total_frames, label = line.split()
                    if self.data_prefix is not None and int(total_frames) > 0:
                        frame_dir = os.path.join(self.data_prefix, frame_dir)
                    video_infos.append(
                        dict(frame_dir=frame_dir,
                             start_index=1,
                             total_frames=int(total_frames),
                             label=int(label)))
        return video_infos

    def sample_indices(self, frame_list: List[int]) -> np.ndarray:
        """TDN official implementation.
        Returns the start index for each segment. One-indexed.
        Will read `[frames[i:i+5] for i in indices]`
        """

        if ((len(frame_list) - self.num_frames + 1) < self.num_segments):
            average_duration = (len(frame_list) - 5 + 1) // (self.num_segments)
        else:
            average_duration = (len(frame_list) - self.num_frames +
                                1) // (self.num_segments)
        offsets = []
        if average_duration > 0:
            offsets += list(
                np.multiply(list(range(self.num_segments)), average_duration) +
                randint(average_duration, size=self.num_segments))
        elif len(frame_list) > self.num_segments:
            if ((len(frame_list) - self.num_frames + 1) >= self.num_segments):
                offsets += list(
                    np.sort(
                        randint(len(frame_list) - self.num_frames + 1,
                                size=self.num_segments)))
            else:
                offsets += list(
                    np.sort(randint(len(frame_list) - 5 + 1, size=self.num_segments)))
        else:
            offsets += list(np.zeros((self.num_segments,)))
        return np.array(offsets).astype(int)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        """TDN getitem. 
        Samples frame indices. And read 5 frames starting from each index.
        
        Returns:
            Tuple[Tensor, int], tensor of shape (num_segments, num_frames, C, H, W)
        """

        frame_list = []
        video_info = self.anno[idx]
        frame_dir = video_info['frame_dir']
        start_index = video_info['start_index'] if self.anno_col == 4 else 1
        total_frames = video_info['total_frames']
        label = video_info['label']
        samples = self.sample_indices(list(range(total_frames))) + start_index
        for i in samples:
            for j in range(self.num_frames):
                frame_path = os.path.join(frame_dir, self.tmpl.format(i + j))
                frame = read_image(frame_path)
                frame_list.append(frame)
        frame_tensor = torch.stack(frame_list, dim=0)
        if self.transform is not None:
            frame_tensor = self.transform(frame_tensor)
        assert frame_tensor.shape[0] == self.num_segments * self.num_frames, \
            f'{frame_tensor.shape} is not [{self.num_segments} * {self.num_frames}, C, H, W]'
        return frame_tensor, label

    def __len__(self) -> int:
        return len(self.anno)


if __name__ == '__main__':
    data_root = 'data'
    anno_path = 'data/Binary/all-val.txt'
    ds = TDNDataset(data_root, anno_path=anno_path, data_prefix=None, num_segments=8)
    print(len(ds))
    l = list(range(10, 20))
    s = ds.sample_indices(l).tolist()
    print(s)

    for i in range(10):
        ds[i]