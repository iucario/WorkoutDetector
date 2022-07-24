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
from torchvision.models.feature_extraction import (create_feature_extractor,
                                                   get_graph_node_names)
from workoutdetector.datasets import (Pipeline, RepcountHelper, build_test_transform)
from workoutdetector.settings import PROJ_ROOT, REPCOUNT_ANNO_PATH
from workoutdetector.trainer import LitModel
from einops import rearrange

onnxruntime.set_default_logger_severity(3)


class Dataset(torch.utils.data.Dataset):
    """DataLoader for one video
    
    Args:
        video_path: path to video
        stride (int): stride of prediction frame indices. Default 1.
        step (int): step of sampling frames. Default 1.
        length (int): length of input frames to model. Default 8.
        input_shape (str): shape of input frames. Default 'TCHW'. 'CTHW' for 3D CNN.
        transform (Callable): transform for frames.
    """

    def __init__(self,
                 video_path: str,
                 stride: int = 1,
                 step: int = 1,
                 length: int = 8,
                 input_shape: str = 'TCHW',
                 transform: Optional[Callable] = None):
        assert step >= 1, 'step must be greater than or equal to 1'
        assert stride >= 1, 'stride must be greater than or equal to 1'

        video, _, meta = read_video(video_path)
        # start index of each inputs.
        self.indices = list(range(0, len(video) - step * length + 1, stride))
        self.shape = ' '.join(list(input_shape))
        self.video = video
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
        t, h, w, c = frames.shape
        frames = rearrange(frames, f'T C H W -> {self.shape}', T=t, C=c, H=h, W=w)
        return frames, i

    def __len__(self) -> int:
        return len(self.indices)


@torch.no_grad()
def infer_one_video(model, path: str, out_path: str, stride: int, step: int, transform,
                    total_frames, reps, class_name, device):
    ds = Dataset(path, stride=stride, step=step, length=8, transform=transform)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
    video_name = path.split('.')[0] + '.mp4'
    res_dict = dict(
        video_name=video_name,
        model='TSM',
        stride=stride,
        step=step,
        length=8,
        fps=ds.fps,
        input_shape=[1, 8, 3, 224, 224],
        checkpoint=ckpt,
        total_frames=total_frames,
        ground_truth=reps,
        action=class_name,
    )
    scores: Dict[int, dict] = dict()
    for x, i in loader:
        start_index = i.item()
        with torch.no_grad():
            pred: Tensor = model(x.to(device))
            scores[start_index] = dict((j, v.item()) for j, v in enumerate(pred[0]))
        # print(scores[start_index])
    res_dict['scores'] = scores

    json.dump(res_dict, open(out_path, 'w'))
    print(f'{video_name} result saved to {out_path}')
    # Save feature maps to pkl


def main(ckpt: str,
         out_dir: str,
         stride: int = 1,
         step: int = 1,
         rank: int = 0,
         world_size: int = 1):
    """Inference videos in the dataset and save results to JSON"""

    device = f'cuda:{rank}'
    data_root = os.path.expanduser('~/data/RepCount')
    helper = RepcountHelper(data_root, os.path.join(data_root, 'annotation.csv'))
    data = helper.get_rep_data(split=['train', 'val', 'test'], action=['all'])
    # data parallel
    part_size = len(data) // world_size
    if rank == world_size - 1:
        end = len(data)
    else:
        end = part_size * (rank + 1)

    transform = build_test_transform(person_crop=False)
    model = LitModel.load_from_checkpoint(ckpt)
    model.to(device)
    model.eval()
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # Feature extraction
    avgpool = nn.AdaptiveAvgPool2d(1)
    feature_maps = []  # item: (batch*8, 2048)

    def hook_feat_map(mod, inp, out):
        o = avgpool(out).view(out.shape[0], -1)
        feature_maps.append(o)

    # model.model.base_model.register_forward_hook(hook_feat_map)

    for item in list(data.values())[rank * part_size:end]:
        out_path = os.path.join(out_dir,
                                f'{item.video_name}.stride_{stride}_step_{step}.json')
        if os.path.exists(out_path):
            print(f'{out_path} already exists. Skip.')
            continue
        infer_one_video(model, item.video_path, out_path, stride, step, transform,
                        item.total_frames, item.reps, item.class_, device)
        # feature_maps_path = os.path.join(
        #     out_dir, f'{item.video_name}.stride_{stride}_step_{step}.pkl')
        # torch.save(feature_maps, feature_maps_path)
        # feature_maps = []


if __name__ == '__main__':
    ckpt = 'checkpoints/repcount-12/best-val-acc=0.841-epoch=26-20220711-191616.ckpt'
    out_dir = f'out/acc_0.841_epoch_26_20220711-191616_1x1'
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default=ckpt)
    parser.add_argument('--out_dir', type=str, default=out_dir)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    args = parser.parse_args()

    main(ckpt,
         out_dir,
         stride=args.stride,
         step=1,
         rank=args.rank,
         world_size=args.world_size)