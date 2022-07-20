import argparse
import json
import os
import os.path as osp
import time
from bisect import bisect_left
from collections import deque
from os.path import join as osj
from typing import Callable, Deque, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import onnx
import onnxruntime
import pandas as pd
import torch
import torchvision.transforms as T
import tqdm
from pytorch_lightning import Trainer
from torch import Tensor, nn
from torchvision.io import VideoReader, read_video
from workoutdetector.datasets import (Pipeline, RepcountHelper, build_test_transform)
from workoutdetector.settings import PROJ_ROOT, REPCOUNT_ANNO_PATH
from workoutdetector.trainer import LitModel

onnxruntime.set_default_logger_severity(3)


class Dataset(torch.utils.data.Dataset):
    """DataLoader for one video
    
    Args:
        video_path: path to video
        stride (int): stride of prediction frame indices. Default 1.
        step (int): step of sampling frames. Default 1.
        length (int): length of input frames to model. Default 8.
        transform (Callable): transform for frames.
    """

    def __init__(self,
                 video_path: str,
                 stride: int = 1,
                 step: int = 1,
                 length: int = 8,
                 transform: Optional[Callable] = None):
        assert step >= 1, 'step must be greater than or equal to 1'
        assert stride >= 1, 'stride must be greater than or equal to 1'

        video, _, meta = read_video(video_path)
        # start index of each inputs.
        self.indices = list(range(0, len(video) - step * length, stride))
        self.video = video.permute(0, 3, 1, 2)  # (T, C, H, W)
        self.meta = meta
        self.fps: float = meta['video_fps']
        self.step = step
        self.length = length
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        """Get a frame and its start frame index."""

        i = self.indices[index]
        frames = self.video[i:i + self.step * self.length:self.step]
        assert frames.shape[0] == self.length
        if self.transform is not None:
            frames = self.transform(frames)
        return frames, i

    def __len__(self) -> int:
        return len(self.indices)


def main(ckpt: str, out_dir: str, stride: int = 1, step: int = 1):
    """Inference videos in the dataset and save results to JSON"""

    data_root = os.path.expanduser('~/data/RepCount')
    helper = RepcountHelper(data_root, os.path.join(data_root, 'annotation.csv'))
    data = helper.get_rep_data(split=['val', 'test'], action=['all'])
    transform = build_test_transform(person_crop=False)
    device = 'cuda:7'
    model = LitModel.load_from_checkpoint(ckpt)
    model.to(device)
    model.eval()
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    for item in data.values():
        out_path = os.path.join(out_dir,
                                f'{item.video_name}.stride_{stride}_step_{step}.json')
        if os.path.exists(out_path):
            print(f'{out_path} already exists. Skip.')
            continue
        ds = Dataset(item.video_path,
                     stride=stride,
                     step=step,
                     length=8,
                     transform=transform)
        loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
        res_dict = dict(
            video_name=item.video_name,
            model='TSM',
            stride=stride,
            step=step,
            length=8,
            fps=ds.fps,
            input_shape=[1, 8, 3, 224, 224],
            checkpoint=ckpt,
            total_frames=item.total_frames,
            ground_truth=item.reps,
            action=item.class_,
        )
        scores: Dict[int, dict] = dict()
        for x, i in tqdm.tqdm(loader):
            start_index = i.item()
            with torch.no_grad():
                pred: Tensor = model(x.to(device))
                scores[start_index] = dict((j, v.item()) for j, v in enumerate(pred[0]))
            # print(scores[start_index])
        res_dict['scores'] = scores

        json.dump(res_dict, open(out_path, 'w'))
        print(f'{item.video_name} result saved to {out_path}')


if __name__ == '__main__':
    ckpt = 'checkpoints/repcount-12/best-val-acc=0.923-epoch=10-20220720-151025.ckpt'
    out_dir = f'out/acc_0.923_epoch_10_20220720-151025_1x2'
    main(ckpt, out_dir, step=2)
