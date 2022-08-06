import json
import math
import os
import os.path as osp
from dataclasses import dataclass
from os.path import join as osj
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.io import read_image

CLASSES = ['situp', 'push_up', 'pull_up', 'jump_jack', 'squat', 'front_raise']


class FrameDataset(torch.utils.data.Dataset):
    """Frame dataset for video. 
    label.txt has the following format:
        `dir/to/video/frames start_index total_frames label`
    
    Labels are built using `scripts/build_label_list.py`

    Note:
        `start_index` should be 1-based.

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
        samples = sample_frames(total_frames, self.num_segments, start_index,
                                self.random)
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
        # TODO: read_video seems to be buggy. Causes seg fault.


class FeatureDataset(torch.utils.data.Dataset):
    """Read JSON file containing action scores and return a time series dataset
    
    dict_keys(['video_name', 'model', 'stride', 'step', 'length', 'fps', 
        'input_shape', 'checkpoint', 'total_frames', 'ground_truth', 'action', 'scores'])

    Args:
        json_dir (str): directory containing json files of action scores
        anno_path (str): path to annotation file
        split (str): train, val, test
        normalize (bool): whether to normalize scores to mean 0 and std 1
        softmax (bool): whether to apply softmax to scores
        action (str): action name
        window (int): window size
        stride (int): stride size
        template (str): template for json file name
        num_classes (int): number of classes. If 3 classes, 0 is no action, 1 and 2 are 
            the two states of action.
    Example:
        >>> json_dir = os.path.expanduser(
        ...     '~/projects/WorkoutDetector/out/acc_0.841_epoch_26_20220711-191616_1x1')
        >>> template = '{}.stride_1_step_1.json'
        >>> anno_path = os.path.expanduser("~/data/RepCount/annotation.csv")
        >>> feat_ds = FeatureDataset(json_dir, anno_path, 'train', 'squat',
        ...                          window=1, stride=1, template=template)
    """

    def __init__(
            self,
            json_dir: str,
            anno_path: str,
            split: str,
            normalize: bool = False,
            action: str = 'all',
            window: int = 100,
            stride: int = 1,  # TODO: deprecate this
            template: str = '{}.stride_1_step_1.json',
            num_classes: int = 3,
            softmax: bool = False) -> None:
        super().__init__()
        self.anno_path = anno_path
        self.json_dir = json_dir
        self.template = template
        self.normalize = normalize
        self.num_classes = num_classes
        self.softmax = softmax
        self.x, self.y = self.load_data(split, action, window, stride)
        self.tensor_x = self.x

    def load_data(self,
                  split: str,
                  action: str,
                  window=100,
                  stride=1) -> Tuple[Tensor, List[int]]:
        """
        Returns:
            x (np.ndarray): [num_sample, window, 12]
            y (np.ndarray): [num_sample,] labels of int
        """
        data = list(
            get_rep_data(self.anno_path,
                         data_root=None,
                         split=[split],
                         action=[action]).values())
        x = Tensor([])  # action scores
        y: List[int] = [
        ]  # labels + 1 no-class = 13 classes. Or can be set to 3.
        for item in data:
            js = json.load(
                open(osj(self.json_dir,
                         self.template.format(item.video_name))))
            start_inds = list(map(int, list(js['scores'].keys())))
            n = len(start_inds)
            class_idx = CLASSES.index(item.class_)
            if self.num_classes == 3:
                class_idx = 0
            item_y = reps_to_label(item.reps, js['total_frames'], class_idx)
            item_x: List[np.ndarray] = []
            for _, v in js['scores'].items():
                item_x.append(np.array(list(v.values())))
            assert len(
                item_x
            ) == n, f'{len(item_x)} != n {n}'  # math.ceil(total_frames // stride)
            assert len(item_y) >= n, item  # total_frames
            tx, y_inds = seq_to_windows(Tensor(np.array(item_x)),
                                        window,
                                        stride,
                                        pad_last=True)
            x = torch.cat((x, tx), dim=0)
            y += [item_y[i] if i < len(item_y) else 0 for i in y_inds]
            # TODO: need tests
        return x, y  # type: ignore

    def hmm_stats(self, x: np.ndarray, y: np.ndarray, cov_type: str = 'diag'):
        """Calculate transition matrix and initial pi and means and covariances
            for hmmlearn.hmm.GaussianHMM

        Args:
            x (np.ndarray): [num_sample, feature_dim]
            y (np.ndarray): [num_sample,] labels, int
            cov_type (str): 'diag' or 'full'
        Returns:
            transition_matrix (np.ndarray): [num_states, num_states]
            pi (np.ndarray): [num_states,]
            means (np.ndarray): [num_states, feature_dim]
            covariances (np.ndarray): [num_states, feature_dim]
        Example:
            >>> hmm_stats = feat_ds.hmm_stats(feat_ds.x.squeeze(1), feat_ds.y)
        """
        assert x.shape[0] == y.shape[0], 'x and y must have the same length'
        unique_labels = np.unique(y)
        for i, label in enumerate(unique_labels):
            y[y == label] = i
        n_states = len(unique_labels)
        if x.shape[1] == 1 and x.ndim == 3:
            x = x.squeeze(1)
        assert x.ndim == 2, f'x {x.shape} must be 2D'
        n_samples, n_feats = x.shape

        # compute pi
        pi = np.zeros(n_states, )
        for i in unique_labels:
            pi[i] = np.sum(y == i) / y.shape[0]

        # compute transition matrix:
        transmat = np.zeros((n_states, n_states))
        for i in range(y.shape[0] - 1):
            transmat[y[i], y[i + 1]] += 1
        # normalize rows of transition matrix:
        divisor = np.sum(transmat, axis=1, keepdims=True)
        divisor[divisor == 0] = 1
        transmat = transmat / divisor

        means = np.zeros((n_states, n_feats))
        for i in range(n_states):
            with np.errstate(divide='ignore'):
                means[i, :] = np.nanmean(x[y == i, :], axis=0)
        means[np.isnan(means)] = 0

        if cov_type == 'full':
            cov = np.zeros((n_states, n_feats, n_feats))
        elif cov_type == 'diag':
            cov = np.zeros((n_states, n_feats))
        else:
            raise ValueError(
                f'cov_type must be "diag" or "full", not {cov_type}')
        with np.errstate(divide='ignore'):
            for i in range(n_states):
                if cov_type == 'full':
                    cov[i, :, :] = np.cov(x[y == i, :].T)
                elif cov_type == 'diag':
                    cov[i, :] = np.std(x[y == i, :], axis=0)

        cov[np.isnan(cov)] = 0
        assert transmat.shape == (n_states,
                                  n_states), f'transmat {transmat.shape}'
        assert pi.shape == (n_states, ), f'pi {pi.shape}'
        return transmat, pi, means, cov

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        x, y = self.tensor_x[index], self.y[index]
        if self.softmax:
            x = F.softmax(x, dim=-1)
        if self.normalize:
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True)
            x = (x - mean) / std
        return x, y

    def __len__(self) -> int:
        return len(self.tensor_x)


def sample_frames(total: int,
                  num: int,
                  offset: int = 0,
                  random: bool = True) -> List[int]:
    """Uniformly sample `num` frames from video, randomly.
    
    Args:
        total: int, total frames, 
        num: int, number of frames to sample
        offset: int, offset from start of video
        random: bool, whether to sample randomly, default True.
            If False, always select the first frame in segments.
    Returns: 
        list of frame indices starting from offset
    Examples:
        >>> sample_frames(total=4, num=8, offset=0, random=False)
        [0, 0, 1, 1, 2, 2, 3, 3]
        >>> sample_frames(total=10, num=8, offset=0, random=False)
        [0, 1, 2, 3, 4, 5, 6, 7]
        >>> sample_frames(total=40, num=8, offset=0, random=False)
        [0, 5, 10, 15, 20, 25, 30, 35]
        >>> sample_frames(total=40, num=8, offset=0, random=True)
        [2, 6, 13, 18, 24, 29, 32, 36]
        >>> sample_frames(total=40, num=8, offset=20, random=False)
        [20, 25, 30, 35, 40, 45, 50, 55]
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
    if random:
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


@dataclass
class RepcountItem:
    """RepCount dataset video item"""

    video_path: str  # the absolute video path
    frames_path: str  # the absolute rawframes path
    total_frames: int
    class_: str
    count: int
    reps: List[int]  # start_1, end_1, start_2, end_2, ...
    split: str
    video_name: str
    fps: float = 30.0
    ytb_id: Optional[str] = None  # YouTube id
    ytb_start_sec: Optional[int] = None  # YouTube start sec
    ytb_end_sec: Optional[int] = None  # YouTube end sec

    def __str__(self):
        return (f'video: {self.video_name}\nclass: {self.class_}\n'
                f'count: {self.count}\nreps: {self.reps}\nfps: {self.fps}\n'
                f'total_frames: {self.total_frames}')

    def __getitem__(self, key):
        return self.__dict__[key]

    def __iter__(self):
        return iter(self.__dict__.items())


def get_rep_data(anno_path: str,
                 data_root: str = None,
                 split: List[str] = ['test'],
                 action: List[str] = ['situp']) -> Dict[str, RepcountItem]:
    """Return the RepCount dataset items
    Args:
        anno_path (str): the annotation file path
        data_root (str): the data root path, e.g. 'data/RepCount'
        split (List[str]): list of the split names
        action (List[str]): list of the action names. If ['all'], all actions are used.

    Returns:
        dict of name: RepcountItem
    """
    assert len(split) > 0, 'split must be specified, e.g. ["train", "val"]'
    assert len(
        action) > 0, 'action must be specified, e.g. ["pull_up", "squat"]'
    data_root = '' if data_root is None else data_root
    split = [x.lower() for x in split]
    action = [x.lower() for x in action]
    if 'all' in action:
        action = CLASSES
    df = pd.read_csv(anno_path, index_col=0)
    df = df[df['split'].isin(split)]
    df = df[df['class_'].isin(action)]
    df = df.reset_index(drop=True)
    ret = {}
    for idx, row in df.iterrows():
        name = row['name']
        name_no_ext = name.split('.')[0]
        class_ = row['class_']
        split_ = row['split']
        video_path = os.path.join(data_root, 'videos', split_, name)
        frame_path = os.path.join(data_root, 'rawframes', split_, name_no_ext)
        total_frames = -1
        if os.path.isdir(
                frame_path):  # TODO: this relies on rawframe dir. Not good.
            total_frames = len(os.listdir(frame_path))
        video_id = row['vid']
        count = int(row['count'])
        if count > 0:
            reps = [int(x) for x in row['reps'].split()]
        else:
            reps = []
        item = RepcountItem(video_path, frame_path, total_frames, class_,
                            count, reps, split_, name, row.fps, video_id,
                            row['start'], row['end'])
        ret[name] = item
    return ret


def get_video_list(anno_path: str,
                   split: str,
                   action: Optional[str] = None,
                   max_reps: int = 2) -> List[dict]:
    """Returns a list of dict of repetitions.

    Args:
        split: str, train or val or test
        action: str, action class name. If none, all actions are used.
        max_reps: int, limit the number of repetitions per video.
            If less than 1, all repetitions are used.

    Returns:
        list of dict: videos, 
            {
                video_path: path to raw frames dir, relative to `root`
                start: start_frame_index, start from 1,
                end: end_frame_index
                length: end_frame_index - start_frame_index + 1
                class: action class,
                label: 0 or 1
            }
    """
    df = pd.read_csv(anno_path)
    if action is not None:
        df = df[df['class_'] == action]
    videos = []
    for row in df.itertuples():
        name = row.name.split('.')[0]
        count = row.count
        if count > 0:
            reps = list(map(int, row.reps.split()))[:max_reps * 2]
            for start, end in zip(reps[0::2], reps[1::2]):
                start += 1  # plus 1 because img index starts from 1
                end += 1  # but annotated frame index starts from 0
                mid = (start + end) // 2
                videos.append({
                    'video_path':
                    os.path.join('RepCount/rawframes', split, name),
                    'start':
                    start,
                    'end':
                    mid,
                    'length':
                    mid - start + 1,
                    'class':
                    row.class_,
                    'label':
                    0
                })
                videos.append({
                    'video_path':
                    os.path.join('RepCount/rawframes', split, name),
                    'start':
                    mid + 1,
                    'end':
                    end,
                    'length':
                    end - mid,
                    'class':
                    row.class_,
                    'label':
                    1
                })
    return videos


def reps_to_label(reps: List[int], total: int, class_idx: int):
    """Spread interval to label.
        Set class_idx to 0 to get 3 classes.
    """
    y = [0] * total
    for start, end in zip(reps[::2], reps[1::2]):
        mid = (start + end) // 2 + 1
        # class index plus 1 because no-class is 0
        y[start:mid] = [class_idx * 2 + 1] * (mid - start)
        y[mid:end + 1] = [class_idx * 2 + 2] * (end + 1 - mid)
    assert len(y) == total, f'len(y) = {len(y)} != total {total}\n{reps}'
    return y


def seq_to_windows(x: Tensor,
                   window: int,
                   stride: int,
                   pad_last: bool = True) -> Tuple[Tensor, List[int]]:
    """Returns windows and corresponding last frame indices."""
    assert x.dim() == 2, x.shape
    assert pad_last, "pad_last=False is not implemented"
    t = torch.zeros(x.shape[0] + window - 1, x.shape[1])
    if pad_last:
        t[:x.shape[0], :] = x
    ret, inds = [], []
    for i in range(0, x.shape[0], stride):
        ret.append(t[i:i + window, :])
        inds.append(i + window - 1)
    assert len(ret) == math.ceil(x.shape[0] / stride), \
        f'{len(ret)} != {math.ceil(x.shape[0] / stride)}'
    return torch.stack(ret), inds


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
        '~/projects/WorkoutDetector/out/acc_0.841_epoch_26_20220711-191616_1x1'
    )
    template = '{}.stride_1_step_1.json'
    anno_path = os.path.expanduser("~/data/RepCount/annotation.csv")

    feat_ds = FeatureDataset(json_dir,
                             anno_path,
                             'train',
                             normalize=False,
                             action='squat',
                             window=10,
                             stride=70,
                             template=template,
                             softmax=True)
    print(len(feat_ds))
    print(feat_ds.x.shape, len(feat_ds.y))
    # hmm_stats = feat_ds.hmm_stats(feat_ds.x.squeeze(1), feat_ds.y)
    print(feat_ds[0][0][0])
