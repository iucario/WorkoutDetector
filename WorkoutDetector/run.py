import argparse
from typing import Tuple
from WorkoutDetector.datasets import RepcountVideoDataset
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

from mmaction.datasets import build_dataset
from mmaction.models import build_model
from mmaction.apis import train_model
import copy
import mmcv
from mmcv import Config
from mmcv.runner import set_random_seed

from mmaction.datasets.base import BaseDataset
from mmaction.datasets.builder import DATASETS

proj_config = yaml.safe_load(
    open(os.path.join(os.path.dirname(__file__), 'utils/config.yml')))
PROJ_ROOT = proj_config['proj_root']


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
        super(MyDataset, self).__init__(ann_file, pipeline, data_prefix, test_mode,
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
                    frame_dir = os.path.join(self.data_prefix, frame_dir)
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


def train(cfg: Config) -> None:
    # Build the dataset
    datasets = [build_dataset(cfg.data.train)]

    # Build the recognizer
    model = build_model(cfg.model, train_cfg=None, test_cfg=dict(average_clips='prob'))

    # Create work_dir
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
    train_model(model, datasets, cfg, distributed=False, validate=True)


def main():
    ACTIONS = [
        'situp', 'push_up', 'pull_up', 'bench_pressing', 'jump_jack', 'squat',
        'front_raise'
    ]
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--action', type=str, default='jump_jack', choices=ACTIONS)
    args = parser.parse_args()

    # configs
    config = os.path.join(PROJ_ROOT, 'WorkoutDetector/tsm_config.py')
    cfg = Config.fromfile(config)

    cfg.setdefault('omnisource', False)

    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.data.train.ann_file = os.path.join(PROJ_ROOT, 'data/Binary', f'{args.action}-train.txt')
    cfg.data.val.ann_file = os.path.join(PROJ_ROOT, 'data/Binary', f'{args.action}-val.txt')
    cfg.data.test.ann_file = os.path.join(PROJ_ROOT, 'data/Binary', f'{args.action}-test.txt')

    # cfg.resume_from = osp.join(cfg.work_dir, 'latest.pth')
    print(cfg.pretty_text)
    
    train(cfg)


if __name__ == '__main__':
    main()