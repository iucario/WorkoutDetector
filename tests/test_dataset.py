from workoutdetector.datasets import ImageDataset, build_dataset
from workoutdetector.datasets.common import FeatureDataset, reps_to_label, seq_to_windows
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
    anno = os.path.expanduser('~/data/Binary/all-train.txt')
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


def test_reps_to_label():
    r1 = [1, 3, 4, 6, 7, 10]
    a1 = [0, 1, 1, 2, 1, 1, 2, 1, 1, 2, 2, 0]
    y1 = reps_to_label(r1, len(a1), class_idx=0)
    assert y1 == a1, f'y1 {y1} != a1 {a1}\nr1={r1}'
    r2 = [0, 1, 6, 7, 8, 10, 10, 14]
    a2 = [1, 2, 0, 0, 0, 0, 1, 2, 1, 1, 1, 1, 1, 2, 2]
    y2 = reps_to_label(r2, len(a2), class_idx=0)
    assert y2 == a2, f'y2 {y2} != a2 {a2}\nr2={r2}'
    r3 = [3, 9]
    a3 = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 0, 0]
    y3 = reps_to_label(r3, len(a3), class_idx=0)
    assert y3 == a3, f'y3 {y3} != a3 {a3}\nr3={r3}'


def test_seq_to_windows():
    x = torch.arange(10).reshape(10, 1) + 1
    w, _ = seq_to_windows(x, window=4, stride=3, pad_last=True)
    assert w.shape == (4, 4, 1), f'{w.shape}'

    w, _ = seq_to_windows(x, window=20, stride=10, pad_last=True)
    assert w.shape == (1, 20, 1), f'{w.shape}'


def test_FeatureDatasest():
    data_root = os.path.expanduser("~/data/RepCount")
    json_dir = 'out/acc_0.841_epoch_26_20220711-191616_1x1'
    anno_path = 'datasets/RepCount/annotation.csv'
    window = 100
    num_classes = 3
    feat_data = FeatureDataset(json_dir,
                               anno_path,
                               'train',
                               normalize=False,
                               num_classes=num_classes,
                               softmax=False,
                               stride=17,
                               window=window)
    assert len(feat_data.tensor_x) == len(feat_data.y), \
        f'x {len(feat_data.tensor_x)} != y {len(feat_data.y)}'
    assert feat_data.tensor_x.shape[-2:] == (window, 12), \
        f'x.shape {feat_data.tensor_x.shape} != (-1, {window}, {num_classes})'


if __name__ == '__main__':
    # test_reps_to_label()
    # test_seq_to_windows()
    test_FeatureDatasest()