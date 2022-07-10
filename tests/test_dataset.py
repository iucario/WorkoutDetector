from workoutdetector.datasets import ImageDataset, build_dataset
import torch
import os
import os.path as osp
from os.path import join as osj


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
