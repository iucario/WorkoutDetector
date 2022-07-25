import os
from os.path import join as osj
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from fvcore.common.config import CfgNode
from torchvision.io import read_video
from workoutdetector.datasets import RepcountHelper, build_test_transform
from workoutdetector.models import build_model
from workoutdetector.utils.torch_infer import Dataset
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import pickle

ckpt_path = 'checkpoints/finetune/ssv2_videomae_pretrain_base_patch16_224_frame_16x2_tube_mask_ratio_0.9_e2400.pth'
cfg = CfgNode(yaml.safe_load(open('workoutdetector/configs/defaults.yaml')))

cfg.model.update({
    'num_frames': 16,
    'num_segments': 1,
    'model_type': 'videomae',
    'checkpoint': ckpt_path,
    'example_input_array': [1, 3, 16, 224, 224],
})
print(cfg)

# data parallel
arg_parser = ArgumentParser()
arg_parser.add_argument('--rank', type=str, default=0)
arg_parser.add_argument('--world_size', type=int, default=1)
args = arg_parser.parse_args()

world_size = args.world_size
device = f'cuda:{args.rank}'

model = build_model(cfg)
model.eval()
model.to(device)

transform = build_test_transform()


def fx_one_video(model,
                 video_path: str,
                 out_path: str,
                 length=16,
                 stride=1,
                 step=1) -> None:
    """Save features of one video to a pickle."""
    ds = Dataset(video_path,
                 stride=stride,
                 step=step,
                 length=length,
                 input_shape='CTHW',
                 transform=transform)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    res = []
    d = dict(video_path=video_path,
             model='VideoMAE',
             stride=stride,
             step=step,
             length=length,
             total_frames=len(ds))
    for x, idx in loader:
        feat = model.forward_features(x.cuda())
        res.append(feat.squeeze(0).cpu().detach().numpy())
    arr = np.stack(res, axis=0)
    d['features'] = arr
    pickle.dump(d, open(out_path, 'wb'))


helper = RepcountHelper(os.path.expanduser('~/data/RepCount'),
                        os.path.expanduser('~/data/RepCount/annotation.csv'))

data = helper.get_rep_data(split=['test'], action=['all'])
out_dir = 'out/videomae_features_length_16_strid_4_step_1'

total = len(data)
start = world_size * int(args.rank)
if args.rank == world_size - 1:
    end = total
else:
    end = total // world_size * (args.rank + 1)
print(f'rank {args.rank} from video index {start} to {end}')
item_list = list(data.values())
os.makedirs(out_dir, exist_ok=True)

for i in range(start, end):
    item = item_list[i]
    video_path = item['video_path']
    out_path = osj(out_dir, os.path.basename(video_path) + '.pkl')
    if os.path.exists(out_path):
        print('Skip', out_path)
        continue
    fx_one_video(model, video_path, out_path, stride=4, step=1, length=16)
    print(f'{i}/{total} {video_path} saved to {out_path}')