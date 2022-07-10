from workoutdetector.datasets import ImageDataset, build_dataset
import torch
import os
import os.path as osp
from os.path import join as osj
from workoutdetector.datasets import build_dataset
from fvcore.common.config import CfgNode
from torch.utils.data import DataLoader


def test_ImageDataset():
    data_root = osp.expanduser('~/data/situp')
    train_anno = osj(data_root, 'train.txt')
    train_set = ImageDataset(data_root,
                             data_prefix='train',
                             anno_path=train_anno,
                             transform=None)
    img, label = train_set[0]
    assert len(train_set)
    assert img.shape[0] == 3, f'{img.shape}'


def test_TDNDataset():
    root = 'data'
    anno = '/home/user/data/Binary/all-train.txt'
    cfg = CfgNode(new_allowed=True)
    cfg.merge_from_file('workoutdetector/configs/tdn.yaml')
    batch = cfg.data.batch_size
    num_seg = cfg.data.num_segments
    num_frames = cfg.data.num_frames
    ds = build_dataset(cfg.data, split='train')
    assert len(ds), f'No data in {ds.root}'
    loader = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=4)
    for _, (x, y) in zip(range(10), loader):
        assert x.shape == (batch, num_seg * num_frames, 3, 224, 224), \
            f'{x.shape} is not ({batch}, {num_seg} * {num_frames}, 3, 224, 224)'
