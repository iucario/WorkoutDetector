import io
from typing import Any, Dict, List, OrderedDict, Union

import cv2
import decord
import matplotlib
import matplotlib.pyplot as plt
import moviepy.editor as mpy
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.collections import LineCollection

CLASSES = ['situp', 'push_up', 'pull_up', 'jump_jack', 'squat', 'front_raise']


def plt_params() -> dict:
    COLORS = list(plt.get_cmap('Set3').colors)
    plt.style.use('seaborn-dark')
    params = {
        'figure.dpi': 300,
        'figure.figsize': (8, 5),
        'figure.autolayout': True,
        'lines.linewidth': 0.8,
        'axes.prop_cycle': plt.cycler('color', COLORS),
        'font.size': 8,
        'font.family': 'serif'
    }
    return params


matplotlib.RcParams.update(plt_params())


def plot_pred(result: List[int],
              gt: List[int],
              total_frames: int,
              info: Dict[str, Any],
              step: int = 8) -> None:
    """Plot segmentation result and ground truth.
    
    Args:
        result (List[int]): segmentation result. [start_1, end_1, start_2, ...]
        gt (List[int]): ground truth. [start_1, end_1, start_2, ...]
        total_frames (int): total number of frames.
        info (Dict[str, Any]): info dict. Inferenced json.
        step (int): step of prediction.
    """

    fps = info.get('fps', 30)
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
                             0.5,
                             color=['C5', 'C4'][i % 4 // 2])
        plt.gca().add_patch(rect)
    for j in range(0, len(result), 2):
        rect = plt.Rectangle((result[j], 0.0), (result[j + 1] - result[j]),
                             0.49,
                             color=['C0', 'C2'][j % 4 // 2])
        plt.gca().add_patch(rect)
    # plt.vlines(result, color='C1', ymin=0.0, ymax=1.0)
    plt.title(f'{info["video_name"]}, {info["action"]}, count={len(gt)//2},'\
        ' Up: ground truth, Down: prediction')
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
    max_num_ticks = 10
    if action_only:
        plt.figure(figsize=(10, 3))
        plt.xlim(0, total_frames)
        plt.ylim(0, 1)
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


def to_softmax(d: Union[Dict[str, float], OrderedDict[str, float]]) -> Dict[str, float]:
    """Raw score to softmax score.
    
    Args:
        d (Dict[str, float]): raw score dict of one frame.
    Returns:
        Dict[str, float]: softmax score dict of one frame.
    """

    softmax_d = F.softmax(torch.Tensor(list(d.values())), dim=0)
    return dict(zip(d.keys(), softmax_d.numpy()))


class Vis3DPose:

    def __init__(self,
                 item,
                 layout='nturgb+d',
                 fps=12,
                 angle=(30, 45),
                 fig_size=(8, 8),
                 dpi=80):
        kp = item['keypoint']
        self.kp = kp
        assert self.kp.shape[-1] == 3
        self.layout = layout
        self.fps = fps
        self.angle = angle  # For 3D data only
        self.colors = ('#3498db', '#000000', '#e74c3c')  # l, m, r
        self.fig_size = fig_size
        self.dpi = dpi

        assert layout == 'nturgb+d'
        if self.layout == 'nturgb+d':
            self.num_joint = 25
            self.links = np.array([(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
                                   (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
                                   (13, 1), (14, 13),
                                   (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                                   (20, 19), (22, 8), (23, 8), (24, 12), (25, 12)],
                                  dtype=np.int) - 1
            self.left = np.array([5, 6, 7, 8, 13, 14, 15, 16, 22, 23], dtype=np.int) - 1
            self.right = np.array([9, 10, 11, 12, 17, 18, 19, 20, 24, 25],
                                  dtype=np.int) - 1
            self.num_link = len(self.links)
        self.limb_tag = [1] * self.num_link

        for i, link in enumerate(self.links):
            if link[0] in self.left or link[1] in self.left:
                self.limb_tag[i] = 0
            elif link[0] in self.right or link[1] in self.right:
                self.limb_tag[i] = 2

        assert len(kp.shape) == 4 and kp.shape[3] == 3 and kp.shape[2] == self.num_joint
        x, y, z = kp[..., 0], kp[..., 1], kp[..., 2]

        min_x, max_x = min(x[x != 0]), max(x[x != 0])
        min_y, max_y = min(y[y != 0]), max(y[y != 0])
        min_z, max_z = min(z[z != 0]), max(z[z != 0])
        max_axis = max(max_x - min_x, max_y - min_y, max_z - min_z)
        mid_x, mid_y, mid_z = (min_x + max_x) / 2, (min_y + max_y) / 2, (min_z +
                                                                         max_z) / 2
        self.min_x, self.max_x = mid_x - max_axis / 2, mid_x + max_axis / 2
        self.min_y, self.max_y = mid_y - max_axis / 2, mid_y + max_axis / 2
        self.min_z, self.max_z = mid_z - max_axis / 2, mid_z + max_axis / 2

        self.images = []

    def get_img(self, dpi=80):
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=dpi)
        buf.seek(0)
        img = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        return cv2.imdecode(img, -1)

    def vis(self):
        self.images = []
        plt.figure(figsize=self.fig_size)
        for t in range(self.kp.shape[1]):
            ax = plt.gca(projection='3d')
            ax.set_xlim3d([self.min_x, self.max_x])
            ax.set_ylim3d([self.min_y, self.max_y])
            ax.set_zlim3d([self.min_z, self.max_z])
            ax.view_init(*self.angle)
            ax.set_aspect('auto')
            for i in range(self.num_link):
                for m in range(self.kp.shape[0]):
                    link = self.links[i]
                    color = self.colors[self.limb_tag[i]]
                    j1, j2 = self.kp[m, t, link[0]], self.kp[m, t, link[1]]
                    if not ((np.allclose(j1, 0) or np.allclose(j2, 0)) and
                            link[0] != 1 and link[1] != 1):
                        ax.plot([j1[0], j2[0]], [j1[1], j2[1]], [j1[2], j2[2]],
                                lw=1,
                                c=color)
            self.images.append(self.get_img(dpi=self.dpi))
            ax.cla()
        return mpy.ImageSequenceClip(self.images, fps=self.fps)


def Vis2DPose(item: dict,
              thre=0.2,
              out_shape=(540, 960),
              layout='coco',
              fps=24,
              video=None):
    assert layout == 'coco'

    kp = item['keypoint']
    if 'keypoint_score' in item:
        kpscore = item['keypoint_score']
        kp = np.concatenate([kp, kpscore[..., None]], -1)

    assert kp.shape[-1] == 3
    img_shape = item.get('img_shape', out_shape)
    kp[..., 0] *= out_shape[1] / img_shape[1]
    kp[..., 1] *= out_shape[0] / img_shape[0]

    total_frames = item.get('total_frames', kp.shape[1])
    assert total_frames == kp.shape[1]

    if video is None:
        frames = [
            np.ones([out_shape[0], out_shape[1], 3], dtype=np.uint8) * 255
            for i in range(total_frames)
        ]
    else:
        vid = decord.VideoReader(video)
        frames = [x.asnumpy() for x in vid]
        frames = [cv2.resize(x, (out_shape[1], out_shape[0])) for x in frames]
        if len(frames) != total_frames:
            frames = [
                frames[int(i / total_frames * len(frames))] for i in range(total_frames)
            ]

    if layout == 'coco':
        edges = [(0, 1, 'f'), (0, 2, 'f'), (1, 3, 'f'), (2, 4, 'f'), (0, 5, 't'),
                 (0, 6, 't'), (5, 7, 'ru'), (6, 8, 'lu'), (7, 9, 'ru'), (8, 10, 'lu'),
                 (5, 11, 't'), (6, 12, 't'), (11, 13, 'ld'), (12, 14, 'rd'),
                 (13, 15, 'ld'), (14, 16, 'rd')]
    color_map = {
        'ru': ((0, 0x96, 0xc7), (0x3, 0x4, 0x5e)),
        'rd': ((0xca, 0xf0, 0xf8), (0x48, 0xca, 0xe4)),
        'lu': ((0x9d, 0x2, 0x8), (0x3, 0x7, 0x1e)),
        'ld': ((0xff, 0xba, 0x8), (0xe8, 0x5d, 0x4)),
        't': ((0xee, 0x8b, 0x98), (0xd9, 0x4, 0x29)),
        'f': ((0x8d, 0x99, 0xae), (0x2b, 0x2d, 0x42))
    }

    for i in range(total_frames):
        for m in range(kp.shape[0]):
            ske = kp[m, i]
            for e in edges:
                st, ed, co = e
                co_tup = color_map[co]
                j1, j2 = ske[st], ske[ed]
                j1x, j1y, j2x, j2y = int(j1[0]), int(j1[1]), int(j2[0]), int(j2[1])
                conf = min(j1[2], j2[2])
                if conf > thre:
                    colors = [
                        x + (y - x) * (conf - thre) / 0.8
                        for x, y in zip(co_tup[0], co_tup[1])
                    ]
                    color = tuple([int(x) for x in colors])
                    frames[i] = cv2.line(frames[i], (j1x, j1y), (j2x, j2y),
                                         color,
                                         thickness=2)
    return mpy.ImageSequenceClip(frames, fps=fps)
