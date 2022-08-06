from typing import Any, Dict, List, OrderedDict, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.collections import LineCollection

CLASSES = ['situp', 'push_up', 'pull_up', 'jump_jack', 'squat', 'front_raise']

params = {
    'figure.dpi': 300,
    'figure.figsize': (8, 5),
    'figure.autolayout': True,
    'lines.linewidth': 1,
    'axes.prop_cycle': plt.cycler('color', list(plt.get_cmap('tab20').colors)),
    'font.size': 8,
    'font.family': 'sans-serif',
}


@matplotlib.rc_context(params)
def plot_raw_output(val_x: np.ndarray, val_y: np.ndarray, pred: np.ndarray,
                    baseline: np.ndarray, class_idx: int, title: str) -> None:
    fig, ax = plt.subplots(4, 1, figsize=(7, 4), dpi=400)
    ax[0].plot(pred, 'C2', label='HMM')
    ax[1].plot(val_y, 'C0', label='true')
    for i in range(4):
        if i != 2:
            ax[i].set_ylim(-0.1, 2.1)
        if i < 3:
            ax[i].set_xticks([])
    ax[0].text(-len(val_y) * 0.15, 1, 'HMM')
    ax[1].text(-len(val_y) * 0.15, 1, 'True')
    ax[2].text(-len(val_y) * 0.15, 1, 'Input')
    ax[3].text(-len(val_y) * 0.15, 1, 'Argmax')
    colors = ['C15'] * val_x.shape[1]
    colors[class_idx * 2:class_idx * 2 + 2] = ['C0', 'C2']
    for i, line in enumerate(val_x.T):
        ax[2].plot(line, color=colors[i], alpha=0.8)

    ax[3].plot(baseline, 'C12', label='Argmax')
    ax[-1].set_xlabel('Frame index')
    ax[0].set_title(title)
    plt.tight_layout()
    plt.show()


@matplotlib.rc_context(params)
def plot_hmm(result: List[int],
             gt: List[int],
             orig_reps: List[int],
             total_frames: int,
             fps: float,
             title: str,
             show: bool = True,
             save_path: str = None) -> None:
    video_len = total_frames / fps
    max_num_ticks = 10
    plt.figure(figsize=(7, 2))
    plt.xlabel('Second')
    plt.yticks([])
    plt.ylim(0, 1)
    offset = total_frames // 10
    plt.xlim(-offset * 1.1, total_frames + 5)
    h = 0.2
    plt.xticks(
        np.linspace(0, total_frames, max_num_ticks),
        np.round(np.linspace(0.0, video_len, max_num_ticks), 2),
    )
    # background
    rect = plt.Rectangle((0, h), total_frames, 0.6, color='w')
    plt.gca().add_patch(rect)
    for i in range(0, len(gt), 2):
        rect = plt.Rectangle((gt[i], 0.6), (gt[i + 1] - gt[i]), h, color='C1')
        plt.gca().add_patch(rect)
    plt.vlines(gt, color='C0', linewidth=2, ymin=0.6, ymax=0.8)
    for j in range(0, len(result), 2):
        rect = plt.Rectangle((result[j], 0.4), (result[j + 1] - result[j]),
                             h - 0.01,
                             color='C2')
        plt.gca().add_patch(rect)
    for i in range(0, len(orig_reps), 2):
        rect = plt.Rectangle((orig_reps[i], 0.2),
                             (orig_reps[i + 1] - orig_reps[i]),
                             h - 0.01,
                             color='C5')
        plt.gca().add_patch(rect)
    plt.title(title)
    plt.text(-offset, 0.65, 'True', color='C0')
    plt.text(-offset, 0.45, 'HMM', color='C2')
    plt.text(-offset, 0.25, 'Argmax', color='C4')
    if show:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()


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


def to_softmax(
        d: Union[Dict[str, float], OrderedDict[str,
                                               float]]) -> Dict[str, float]:
    """Raw score to softmax score.
    
    Args:
        d (Dict[str, float]): raw score dict of one frame.
    Returns:
        Dict[str, float]: softmax score dict of one frame.
    """

    softmax_d = F.softmax(torch.Tensor(list(d.values())), dim=0)
    return dict(zip(d.keys(), softmax_d.numpy()))


@matplotlib.rc_context(params)
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


@matplotlib.rc_context(params)
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
    gt_class_idx = CLASSES.index(info['action'])
    colors = list(plt.get_cmap('Set3').colors)
    max_num_ticks = 10
    plt.plot(yarr, marker='.', linestyle='None')
    plt.xticks(np.linspace(0, total_frames, max_num_ticks),
               np.round(np.linspace(0.0, video_len, max_num_ticks), 1))
    plt.xlabel('Seconds')
    plt.ylabel('Softmax score')
    plt.title(f"{info['video_name']} {info['action']} count={counts}")
    plt.ylim(0, 1.1)
    plt.vlines(x=gt_reps[0::2] // stride,
               color=colors[gt_class_idx * 2],
               ymin=0.51,
               ymax=1.0)
    plt.vlines(x=gt_reps[1::2] // stride,
               color=colors[gt_class_idx * 2 + 1],
               ymin=0.0,
               ymax=0.49)
    plt.legend(np.array(CLASSES).repeat(2))

    # Indicator
    segs = []
    height = 1.01
    for i in range(len(gt_reps[::2])):
        start = gt_reps[i * 2]
        end = gt_reps[i * 2 + 1]
        mid = (start + end) // 2
        segs.append([(start // stride, height), (mid // stride, height)])
        segs.append([(mid // stride, height), (end // stride, height)])
    lc = LineCollection(
        segs,
        colors=[colors[gt_class_idx * 2], colors[gt_class_idx * 2 + 1]],
        linewidths=1)
    plt.gca().add_collection(lc)
    plt.show()


@matplotlib.rc_context(params)
def plot_per_action(info: dict,
                    softmax: bool = False,
                    action_only: bool = False) -> None:
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
