import argparse
from typing import Tuple
# from WorkoutDetector.datasets import RepcountVideoDataset
import torch
from torch.utils.data import DataLoader
import os
from torch import optim, nn, utils, Tensor
import pytorch_lightning as pl
import torchvision.transforms as T
import timm
import yaml
import einops
import time

from mmaction.models.backbones import ResNetTSM
from mmaction.models.heads import TSMHead

proj_config = yaml.safe_load(
    open(os.path.join(os.path.dirname(__file__), 'utils/config.yml')))
proj_root = proj_config['proj_root']

data_transforms = {
    'train':
        T.Compose([
            T.ToPILImage(),
            T.Resize(256),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    'val':
        T.Compose([
            T.ToPILImage(),
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
}


def get_data_loaders(action: str,
                     batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    data_root = os.path.join(proj_root, 'data')
    train_set = RepcountVideoDataset(root=data_root,
                                     action=action,
                                     split='train',
                                     transform=data_transforms['train'])
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)
    val_set = RepcountVideoDataset(root=data_root,
                                   action=action,
                                   split='val',
                                   transform=data_transforms['val'])
    val_loader = DataLoader(val_set,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True)
    test_set = RepcountVideoDataset(root=data_root,
                                    action=action,
                                    split='test',
                                    transform=data_transforms['val'])
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True)
    return train_loader, val_loader, test_loader


config = '../mmaction2/configs/recognition/tsm/tsm_my_config.py'
checkpoint = 'https://download.openmmlab.com/mmaction/recognition/tsm/'\
    'tsm_r50_1x1x8_50e_sthv2_rgb/tsm_r50_256h_1x1x8_50e_sthv2_rgb_20210816-032aa4da.pth'

from mmcv import Config
import os.path as osp

cfg = Config.fromfile(config)

from mmcv.runner import set_random_seed

DATASET = '/home/umi/projects/WorkoutDetector/data/Binary/'

# optimizer
cfg.optimizer.lr = 0.0015 / 8  # this lr is used for 8 gpus
# learning policy
# cfg.lr_config = dict(policy='step', step=[10, 20])

# Modify dataset type and path
cfg.dataset_type = 'RawframeDataset'
cfg.data_root = DATASET
cfg.data_root_val = DATASET
cfg.ann_file_train = DATASET + 'squat-train.txt'
cfg.ann_file_val = DATASET + 'squat-val.txt'
cfg.ann_file_test = DATASET + 'squat-test.txt'

cfg.data.test.ann_file = cfg.ann_file_test
cfg.data.test.data_prefix = DATASET

cfg.data.train.ann_file = cfg.ann_file_train
cfg.data.train.data_prefix = cfg.data_root

cfg.data.val.ann_file = cfg.ann_file_val
cfg.data.val.data_prefix = cfg.data_root_val

cfg.setdefault('omnisource', False)

cfg.model.cls_head.num_classes = 2

cfg.data.videos_per_gpu = max(1, cfg.data.videos_per_gpu // 8)
cfg.total_epochs = 10
cfg.load_from = checkpoint
cfg.checkpoint_config.interval = 5

cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)

# Save the best
cfg.evaluation.save_best = 'auto'

cfg.log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        # dict(type='WandbLoggerHook',
        #      init_kwargs=dict(project='binary-video-classification', config={**cfg})),
    ])
cfg.work_dir = f'./work_dirs/tsm_8_binary_squat_{time.strftime("%Y%m%d-%H%M%S")}'
# cfg.resume_from = osp.join(cfg.work_dir, 'latest.pth')

# print(cfg.pretty_text)

cfg.train_pipeline = cfg.val_pipeline

from mmaction.datasets import build_dataset
from mmaction.models import build_model
from mmaction.apis import train_model

import mmcv

# Build the dataset
datasets = [build_dataset(cfg.data.train)]

# Build the recognizer
model = build_model(cfg.model,
                    train_cfg=cfg.get('train_cfg'),
                    test_cfg=cfg.get('test_cfg'))

# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_model(model, datasets, cfg, distributed=False, validate=True)