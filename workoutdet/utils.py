from typing import Any, Dict, List, OrderedDict, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.collections import LineCollection

CLASSES = ['situp', 'push_up', 'pull_up', 'jump_jack', 'squat', 'front_raise']

import base64
import math
import os
import os.path as osp
from dataclasses import dataclass
from os.path import join as osj
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torchvision.datasets.utils import (download_and_extract_archive, verify_str_arg)


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


def eval_count(preds: List[int], targets: List[int]) -> Tuple[float, float]:
    """Evaluate count prediction. By mean absolute error and off-by-one error."""

    mae = 0.0
    off_by_one = 0.0
    for pred, target in zip(preds, targets):
        mae += abs(pred - target)
        off_by_one += (abs(pred - target) == 1)
    return mae / len(preds), off_by_one / len(preds)


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
    assert len(action) > 0, 'action must be specified, e.g. ["pull_up", "squat"]'
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
        if os.path.isdir(frame_path):  # TODO: this relies on rawframe dir. Not good.
            total_frames = len(os.listdir(frame_path))
        video_id = row['vid']
        count = int(row['count'])
        if count > 0:
            reps = [int(x) for x in row['reps'].split()]
        else:
            reps = []
        item = RepcountItem(video_path, frame_path, total_frames, class_, count, reps,
                            split_, name, row.fps, video_id, row['start'], row['end'])
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
                    'video_path': os.path.join('RepCount/rawframes', split, name),
                    'start': start,
                    'end': mid,
                    'length': mid - start + 1,
                    'class': row.class_,
                    'label': 0
                })
                videos.append({
                    'video_path': os.path.join('RepCount/rawframes', split, name),
                    'start': mid + 1,
                    'end': end,
                    'length': end - mid,
                    'class': row.class_,
                    'label': 1
                })
    return videos


class RepcountDataset(torch.utils.data.Dataset):
    """Repcount dataset
    https://github.com/SvipRepetitionCounting/TransRAC
    
    Args:
        root: str, root dir
        split: str, train or val or test
    
    Properties:
        classes: list of str, class names
        df: pandas.DataFrame, annotation data
        split: str, train or val or test
        transform: callable, transform for rawframes
    
    Notes:
        File tree::

            |- RepCount
            |   |- rawframes
            |   |   |- train
            |   |   |     |- video_name/img_00001.jpg
            |   |   |- val
            |   |   |- test
            |   |- videos
            |       |- train
            |       ...

        The annotation csv file has columns:
            name: video name, e.g. 'video_1.mp4'
            class: action class name, e.g. 'squat'
            vid: YouTube video id of length 11
            start: start frame index
            end: end frame index
            count: repetition count
            reps: repetition indices, in format `[s1, e1, s2, e2, ...]`

        The annotation csv file is expected to be in:
        `{PROJ_ROOT}/datasets/RepCount/annotations.csv`

    """
    _URL_VIDEO = 'https://1drv.ms/u/s!AiohV3HRf-34ipk0i1y2P1txpKYXFw'
    _URL_ANNO = 'https://1drv.ms/f/s!AiohV3HRf-34i_V9MWtdu66tCT2pGQ'
    _URL_RAWFRAME = 'https://1drv.ms/u/s!AiohV3HRf-34ipwACYfKSHhkZzebrQ'


def parse_json_score(scores: List[dict], softmax: bool = False) -> np.ndarray:
    """Parse JSON score and return numpy array."""
    arr = []
    if softmax:
        score_list: List[Dict[str, float]] = scores
        od_list: List[Dict[str, float]] = []
        for d in score_list:
            od_list.append(to_softmax(d))
    else:
        od_list = scores
    for item in od_list:
        arr.append([item[str(j)] if str(j) in item else 0 for j in range(12)])
    return np.asarray(arr)


def to_softmax(d: Union[Dict[str, float], OrderedDict[str, float]]) -> Dict[str, float]:
    """Raw score to softmax score.
    
    Args:
        d (Dict[str, float]): raw score dict of one frame.
    Returns:
        Dict[str, float]: softmax score dict of one frame.
    """

    softmax_d = F.softmax(torch.Tensor(list(d.values())), dim=0)
    return dict(zip(d.keys(), softmax_d.numpy()))


def plt_params() -> dict:
    COLORS = list(plt.get_cmap('Tab20').colors)
    plt.style.use('seaborn-dark')
    params = {
        'figure.dpi': 300,
        'figure.figsize': (7, 4),
        'figure.autolayout': True,
        'lines.linewidth': 1,
        'axes.prop_cycle': plt.cycler('color', COLORS),
        'font.size': 8,
        'font.family': 'sans-serif',
    }
    return params


def plot_pred(result: List[int],
              gt: List[int],
              total_frames: int,
              fps: float,
              title: str,
              step: int = 8) -> None:
    """Plot segmentation result and ground truth.
    
    Args:
        result (List[int]): segmentation result. [start_1, end_1, start_2, ...]
        gt (List[int]): ground truth. [start_1, end_1, start_2, ...]
        total_frames (int): total number of frames.
        fps (float): fps of video.
        title (str): title of plot.
        step (int): step of prediction.
    """

    video_len = total_frames / fps
    max_num_ticks = 10
    plt.figure(figsize=(8, 2))
    plt.xlabel('Seconds')
    plt.yticks([])
    plt.ylim(0, 1)
    plt.xlim(0, total_frames)
    plt.xticks(np.linspace(0, total_frames, max_num_ticks),
               np.round(np.linspace(0.0, video_len, max_num_ticks), 2))
    for i in range(0, len(gt), 2):
        rect = plt.Rectangle((gt[i], 0.5), (gt[i + 1] - gt[i]),
                             0.2,
                             color=['C5', 'C4'][i % 4 // 2])
        plt.gca().add_patch(rect)
    for j in range(0, len(result), 2):
        rect = plt.Rectangle((result[j], 0.3), (result[j + 1] - result[j]),
                             0.19,
                             color=['C0', 'C2'][j % 4 // 2])
        plt.gca().add_patch(rect)
    # plt.vlines(result, color='C1', ymin=0.0, ymax=1.0)
    plt.title(title)
    plt.show()


def plot_all(gt_reps: np.ndarray,
             info: Dict[str, Any],
             softmax: bool = False,
             stride: int = 8) -> None:
    """
    Args:
        gt_reps (np.ndarray): ground truth reps.
        info (Dict[str, Any]): info dict. Inferenced json for each video.
        softmax (boo): Apply softmax or not.
        stride (int): Predict every stride frames.
    """

    gt_reps = np.array(gt_reps)
    total_frames = info['total_frames']
    fps = info.get('fps', 30)
    video_len = total_frames / fps
    ys = []
    if softmax:
        score_list: List[Dict[str, float]] = list(info['scores'].values())
        od_list: List[Dict[str, float]] = []
        for d in score_list:
            od_list.append(to_softmax(d))
    else:
        od_list = list(info['scores'].values())
    for item in od_list:
        ys.append([item[str(j)] if str(j) in item else 0 for j in range(12)])
    yarr = np.asarray(ys)
    counts = len(gt_reps) // 2
    GT_CLASS_INDEX = CLASSES.index(info['action'])
    COLORS = list(plt.get_cmap('Set3').colors)
    max_num_ticks = 10
    plt.plot(yarr, marker='.', linestyle='None')
    plt.xticks(np.linspace(0, total_frames, max_num_ticks),
               np.round(np.linspace(0.0, video_len, max_num_ticks), 1))
    plt.xlabel('Seconds')
    plt.ylabel('Softmax score')
    plt.title(f"{info['video_name']} {info['action']} count={counts}")
    plt.ylim(0, 1.1)
    plt.vlines(x=gt_reps[0::2] // stride,
               color=COLORS[GT_CLASS_INDEX * 2],
               ymin=0.51,
               ymax=1.0)
    plt.vlines(x=gt_reps[1::2] // stride,
               color=COLORS[GT_CLASS_INDEX * 2 + 1],
               ymin=0.0,
               ymax=0.49)
    plt.legend(np.array(CLASSES).repeat(2))

    # Indicator
    segs = []
    HEIGHT = 1.01
    for i in range(len(gt_reps[::2])):
        start = gt_reps[i * 2]
        end = gt_reps[i * 2 + 1]
        mid = (start + end) // 2
        segs.append([(start // stride, HEIGHT), (mid // stride, HEIGHT)])
        segs.append([(mid // stride, HEIGHT), (end // stride, HEIGHT)])
    lc = LineCollection(
        segs,
        colors=[COLORS[GT_CLASS_INDEX * 2], COLORS[GT_CLASS_INDEX * 2 + 1]],
        linewidths=1)
    plt.gca().add_collection(lc)
    plt.show()


def plot_per_action(info: dict, softmax: bool = False, action_only: bool = False) -> None:
    """Plot prediction for each action.
    
    Args:
        info (dict): info dict. Inferenced json.
        softmax (bool): Apply softmax or not.
        action_only (bool): Plot only predicted action. If true, only
            the action with the highest score will be plotted. This uses
            the entire video and is offline.
            I just take the ground truth label for now.
            #TODO: implement real action detection.
    """
    total_frames = info['total_frames']
    fps = info.get('fps', 30)
    video_len = total_frames / fps
    yarr = parse_json_score(list(info['scores'].values()), softmax)
    max_num_ticks = 10
    if action_only:
        plt.figure(figsize=(8, 2))
        plt.xlim(0, total_frames)
        plt.ylim(-0.1, 1.1)
        idx = CLASSES.index(info['action'])
        plt.plot(yarr[:, idx * 2:idx * 2 + 2])
        plt.xticks(np.linspace(0, total_frames, max_num_ticks),
                   np.round(np.linspace(0.0, video_len, max_num_ticks), 1))
        plt.title(f'{CLASSES[idx]}')
    else:
        fig, ax = plt.subplots(len(CLASSES), 1, figsize=(8, 8))
        for idx in range(len(CLASSES)):
            ax[idx].set_ylim(0, 1.1)
            ax[idx].plot(yarr[:, idx * 2:idx * 2 + 2])
            ax[idx].set_xlim(0, total_frames)
            ax[idx].set_xticks(
                np.linspace(0, total_frames, max_num_ticks),
                np.round(np.linspace(0.0, video_len, num=max_num_ticks), 2))
            ax[idx].set_title(f'{CLASSES[idx]}', y=0.95)
    plt.xlabel('Seconds')
    plt.ylabel('Softmax score')
    plt.show()
