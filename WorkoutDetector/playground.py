import argparse
from typing import Tuple
from WorkoutDetector.datasets import RepcountVideoDataset
from WorkoutDetector.utils import PROJ_ROOT
import torch
import torchvision
from torch.utils.data import DataLoader
import os
from torch import optim, nn, utils, Tensor
import pytorch_lightning as pl
import torchvision.transforms as T
import timm
import yaml
import einops
import time

from mmcv import Config

from mmaction.models.backbones import ResNetTSM, resnet
from mmaction.models.heads import TSMHead
from mmaction.models import build_model

mm_resnet = resnet.ResNet(depth=50)
torch_resnet = torchvision.models.resnet50()

print(mm_resnet, file=open('mm_resnet.txt', 'w'))
print('=' * 30)
print(torch_resnet, file=open('torch_resnet.txt', 'w'))

backbone = ResNetTSM(
    depth=50,
    norm_eval=False,
    shift_div=8,
    pretrained=f'torchvision://resnet50',
)
head = TSMHead(
    num_classes=2,
    in_channels=2048,
    num_segments=8,
    loss_cls=dict(type='CrossEntropyLoss'),
    spatial_type='avg',
    consensus=dict(type='AvgConsensus', dim=1),
    dropout_ratio=0.5,
    init_std=0.001,
    is_shift=True,
    temporal_pool=False,
)
config = os.path.join(PROJ_ROOT, 'WorkoutDetector/tsm_config.py')
cfg = Config.fromfile(config)
mm_model = build_model(cfg.model, train_cfg=None, test_cfg=dict(average_clips='prob'))

print(mm_model, file=open('mm_model.txt', 'w'))

tsm_ckpt = '/home/umi/.cache/torch/hub/checkpoints/tsm_r50_256h_1x1x8_50e_sthv2_rgb_20210816-032aa4da.pth'

tsm_state = torch.load(tsm_ckpt)
meta = tsm_state['meta']
state_dict = tsm_state['state_dict']

with open('tsm_state_dict.txt', 'w') as f:
    for k, v in state_dict.items():
        f.write(f'{k}:\t{v.shape}\n')

with open('mm_model_state_dict.txt', 'w') as f:
    for k, v in mm_model.state_dict().items():
        f.write(f'{k}:\t{v.shape}\n')

# modify cls head
state_dict['cls_head.fc_cls.weight'] = mm_model.state_dict()['cls_head.fc_cls.weight']
state_dict['cls_head.fc_cls.bias'] = mm_model.state_dict()['cls_head.fc_cls.bias']

mm_model.load_state_dict(state_dict)

dummy_x = torch.randn(4, 8, 3, 224, 224)
label = torch.randint(0, 2, (4,))
print(f'{label=}')
dummy_y = mm_model(dummy_x, label=label)
print(f'{dummy_y=}')

for i, k in enumerate(mm_model.state_dict().keys()):
    if i + 2 >= len(state_dict.keys()):
        break
    bk = list(backbone.state_dict().keys())[i]
    v = mm_model.state_dict()[k]
    backbone.state_dict()[bk] = v

x = einops.rearrange(dummy_x, 'N S C H W -> (N S) C H W')
y = backbone(x)
print(f'After backbone {y.shape=}')

z = head(y, num_segs=8)
print(f'After head {z=}')

p = z.argmax(dim=1)
print(f'After argmax {p=}')