import json
import math
import os
import os.path as osp
from os.path import join as osj
from typing import Callable, Dict, List, Optional, Tuple

import einops
import numpy as np
import torch
from torch import Tensor
from torchvision.io import read_image, read_video
from workoutdetector.datasets import RepcountHelper
from workoutdetector.datasets.transform import sample_frames


class FrameDataset(torch.utils.data.Dataset):
    """Frame dataset for video. 
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
        filename_tmpl (str): template of the frame filename, e.g. `img_{:05}.jpg`
        transform (callable, optional): transform to apply to the frames
        anno_col (int): columns in the annotation file, default is 4
            3-column annotation file:
                `frame_dir total_frames label`
            4-column annotation file:
                `frame_dir start_index total_frames label`
        is_test (bool): randomly sampling or not. If true, not random.

    Returns:
        Tensor, shape (N, C, H, W)
        List[int], label
    
    Example:
        >>> data_root = '/data/RepCount/rawframes'
        >>> anno_path = '/data/relabeled/pull_up/train.txt'
        >>> dataset = FrameDataset(data_root,
        ...                    anno_path=anno_path,
        ...                    data_prefix=None,
        ...                    num_segments=8)
    """

    def __init__(self,
                 data_root: str,
                 anno_path: str,
                 data_prefix: Optional[str] = None,
                 num_segments: int = 8,
                 filename_tmpl='img_{:05}.jpg',
                 transform: Optional[Callable] = None,
                 anno_col: int = 4,
                 is_test: bool = False) -> None:
        super().__init__()
        assert osp.isfile(anno_path), f'{anno_path} is not file'
        self.data_root = data_root
        self.data_prefix = osj(data_root, data_prefix if data_prefix else '')
        self.transform = transform
        self.num_segments = num_segments
        self.tmpl = filename_tmpl
        self.anno_col = anno_col
        self.video_infos: List[dict] = self.load_annotation(anno_path)
        self.random = not is_test

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

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        frame_list = []
        video_info = self.video_infos[idx]
        frame_dir = video_info['frame_dir']
        start_index = video_info['start_index'] if self.anno_col == 4 else 1
        total_frames = video_info['total_frames']
        label = video_info['label']
        samples = sample_frames(total_frames, self.num_segments, start_index, self.random)
        for i in samples:
            frame_path = os.path.join(frame_dir, self.tmpl.format(i))
            frame = read_image(frame_path)
            frame_list.append(frame)
        frame_tensor = torch.stack(frame_list, 0)
        if self.transform is not None:
            frame_tensor = self.transform(frame_tensor)
        assert frame_tensor.shape[0] == self.num_segments, \
            f'frame_list.shape[0] = {frame_tensor.shape[0]}, ' \
            f'but self.num_segments = {self.num_segments}'
        return frame_tensor, label

    def __len__(self) -> int:
        return len(self.video_infos)


class VideoDataset(torch.utils.data.Dataset):
    """Video dataset. 
    label.txt has the following format:
        `dir/to/video start_index length label`
    """

    def __init__(self):
        super().__init__()
        pass


class ImageDataset(torch.utils.data.Dataset):
    """General image dataset
    label text files of format `path/to/image.png class` are required

    Args:
        data_root (str): final path will be `data_root/data_prefix/path/to/image.png`
        data_prefix (str): will be appended to data_root
        anno_path (str): absolute annotation path
        transform (Optional[Callable]):, data transform
    """

    def __init__(self,
                 data_root: str,
                 data_prefix: Optional[str] = None,
                 anno_path: str = 'train.txt',
                 transform: Optional[Callable] = None) -> None:
        super().__init__()
        assert osp.isfile(anno_path), f'{anno_path} is not file'
        self.data_prefix = osj(data_root, data_prefix if data_prefix else '')
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


class FeatureDataset(torch.utils.data.Dataset):
    """Read JSON file containing action scores and return a time series dataset
    
    dict_keys(['video_name', 'model', 'stride', 'step', 'length', 'fps', 
        'input_shape', 'checkpoint', 'total_frames', 'ground_truth', 'action', 'scores'])

    Example:
        >>> json_dir = os.path.expanduser(
        ...     '~/projects/WorkoutDetector/out/acc_0.841_epoch_26_20220711-191616_1x1')
        >>> template = '{}.stride_1_step_1.json'
        >>> anno_path = os.path.expanduser("~/data/RepCount/annotation.csv")
        >>> feat_ds = FeatureDataset(json_dir, anno_path, 'train', 'squat',
        ...                          window=1, stride=1, template=template)
    """

    def __init__(self,
                 json_dir: str,
                 anno_path: str,
                 split: str,
                 action: str = 'all',
                 window: int = 100,
                 stride: int = 20,
                 template: str = '{}.stride_1_step_1.json') -> None:
        super().__init__()
        self.helper = RepcountHelper('', anno_path)
        self.classes = self.helper.classes
        self.json_dir = json_dir
        self.template = template
        self.x, self.y = self.load_data(split, action, window, stride)

    def reps_to_label(self, reps, total, classname):
        class_idx = self.classes.index(classname)
        y = [0] * total
        for start, end in zip(reps[::2], reps[1::2]):
            mid = (start + end) // 2
            y[start:mid] = [class_idx * 2 + 1] * (mid - start)
            y[mid:end] = [class_idx * 2 + 2] * (end - mid)  # plus 1 because no-class is 0
        return y

    def load_data(self,
                  split: str,
                  action: str,
                  window=100,
                  stride=20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            x (np.ndarray): [num_sample, window, 12]
            y (np.ndarray): [num_sample,] labels of int
        """
        data = list(self.helper.get_rep_data([split], [action]).values())
        x = []  # action scores
        y: List[int] = []  # labels + 1 no-class = 13 classes
        for item in data:
            js = json.load(open(osj(self.json_dir,
                                    self.template.format(item.video_name))))
            start_ids = list(map(int, list(js['scores'].keys())))
            n = len(start_ids)
            item_y = self.reps_to_label(item.reps, start_ids[-1] + 8, item.class_)
            length = (n - window + 1) // stride
            start_ids = start_ids[:n - window + 1:stride]
            item_x = []
            for i, v in js['scores'].items():
                item_x.append(np.array(list(v.values())))
            assert len(item_x) == n
            assert len(item_y) >= n, item
            x += [item_x[i:i + window] for i in start_ids if i + window <= n]
            # Last frame label is the sequence label
            y += [item_y[i + window - 1] for i in start_ids if i + window <= n]
        x = np.stack(x, axis=0)
        y = np.array(y) # type: ignore
        return x, y # type: ignore

    def hmm_stats(self, x, y):
        """Calculate transition matrix and initial pi and means and covariances
            for hmmlearn.hmm.GaussianHMM

        Args:
            x (np.ndarray): [num_sample, feature_dim]
            y (np.ndarray): [num_sample,] labels of int
        Returns:
            transition_matrix (np.ndarray): [num_states, num_states]
            pi (np.ndarray): [num_states,]
            means (np.ndarray): [feature_dim,]
            covariances (np.ndarray): [feature_dim,]
        Example:
            >>> hmm_stats = feat_ds.hmm_stats(feat_ds.x.squeeze(1), feat_ds.y)
        """
        assert x.shape[0] == y.shape[0], 'x and y must have the same length'
        max_labels = np.arange(np.max(y) + 1)
        n_states = np.max(y) + 1
        n_samples, n_feats = x.shape

        # compute pi
        pi = np.zeros((n_states,))
        for i, u_label in enumerate(max_labels):
            pi[i] = np.count_nonzero(y == u_label)
        # normalize prior probabilities
        pi = pi / pi.sum()

        # compute transition matrix:
        transmat = np.zeros((n_states, n_states))
        for i in range(y.shape[0] - 1):
            transmat[int(y[i]), int(y[i + 1])] += 1
        # normalize rows of transition matrix:
        divisor = np.sum(transmat, axis=1, keepdims=True)
        divisor[divisor == 0] = 1
        transmat = transmat / divisor

        means = np.zeros((n_states, n_feats))
        for i in range(n_states):
            with np.errstate(divide='ignore'):
                means[i, :] = np.nanmean(x[y == i, :], axis=0)
        means[np.isnan(means)] = 0

        cov = np.zeros((n_states, n_feats))
        with np.errstate(divide='ignore'):
            for i in range(n_states):
                # cov[i, :, :] = np.cov(x[y == i, :].T)
                # use line above if HMM using full gaussian distributions are to be used
                cov[i, :] = np.std(x[y == i, :], axis=0)
        cov[np.isnan(cov)] = 0

        return pi, transmat, means, cov

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        return self.x[index], self.y[index]

    def __len__(self) -> int:
        return len(self.x)


if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    # data_root = osp.expanduser('~/data/RepCount/rawframes')
    # anno_path = osp.expanduser('data/relabeled/pull_up/train.txt')
    # dataset = FrameDataset(data_root,
    #                        anno_path=anno_path,
    #                        data_prefix=None,
    #                        num_segments=8)
    # print(len(dataset))
    # random_index = np.random.randint(0, len(dataset))
    # img, label = dataset[random_index]
    # plt.figure(figsize=(8, 4), dpi=200)
    # img = einops.rearrange(img, '(b1 b2) c h w -> (b1 h) (b2 w) c', b2=4)
    # plt.title(f'label: {label}')
    # print(img.shape)
    # plt.imshow(img)
    # plt.show()

    json_dir = os.path.expanduser(
        '~/projects/WorkoutDetector/out/acc_0.841_epoch_26_20220711-191616_1x1')
    template = '{}.stride_1_step_1.json'
    anno_path = os.path.expanduser("~/data/RepCount/annotation.csv")

    feat_ds = FeatureDataset(json_dir,
                             anno_path,
                             'train',
                             'squat',
                             window=1,
                             stride=1,
                             template=template)
    print(len(feat_ds))
    print(feat_ds.x.shape, feat_ds.y.shape)

    hmm_stats = feat_ds.hmm_stats(feat_ds.x.squeeze(1), feat_ds.y)
    print(hmm_stats)