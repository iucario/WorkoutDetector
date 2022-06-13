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


import copy
import os.path as osp

import mmcv

from mmaction.datasets.base import BaseDataset
from mmaction.datasets.builder import DATASETS


@DATASETS.register_module()
class MyDataset(BaseDataset):
    """
    Note:
        label.txt has the following format:
            `dir/to/video/frames start_index total_frames label`
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_prefix=None,
                 test_mode=False,
                 modality='RGB',
                 filename_tmpl='img_{:05}.jpg'):
        super(MyDataset, self).__init__(ann_file,
                                        pipeline,
                                        data_prefix,
                                        test_mode,
                                        modality)

        self.filename_tmpl = filename_tmpl
        self.data_prefix = data_prefix
        print("data_prefix: ", data_prefix)

    def load_annotations(self):
        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                if line.startswith("directory"):
                    continue
                frame_dir, start_index, total_frames, label = line.split()
                if self.data_prefix is not None:
                    frame_dir = osp.join(self.data_prefix, frame_dir)
                video_infos.append(
                    dict(frame_dir=frame_dir,
                         start_index=int(start_index),
                         total_frames=int(total_frames),
                         label=int(label)))
        return video_infos

    def prepare_train_frames(self, idx):
        results = copy.deepcopy(self.video_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        results = copy.deepcopy(self.video_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        return self.pipeline(results)


from mmcv import Config
from mmcv.runner import set_random_seed

config = 'WorkoutDetector/tsm_config.py'
cfg = Config.fromfile(config)

cfg.setdefault('omnisource', False)

cfg.seed = 0
set_random_seed(0, deterministic=False)

# cfg.resume_from = osp.join(cfg.work_dir, 'latest.pth')
print(cfg.pretty_text)
cfg.train_pipeline = cfg.val_pipeline
myset = MyDataset(ann_file=cfg.ann_file_train,
                  pipeline=cfg.train_pipeline,
                  data_prefix=cfg.data_root_train,
                  test_mode=False,
                  filename_tmpl='img_{:05}.jpg')

from mmaction.datasets import build_dataset
from mmaction.models import build_model
from mmaction.apis import train_model

import mmcv

# Build the dataset
datasets = [build_dataset(cfg.data.train)]

# Build the recognizer
model = build_model(cfg.model, train_cfg=None, test_cfg=dict(average_clips='prob'))
# exit(1)
# Create work_dir
mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
train_model(model, datasets, cfg, distributed=False, validate=True)