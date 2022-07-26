import torch
from torch import nn, Tensor
import numpy as np
from workoutdetector.models.bsn import TEM, PEM
from workoutdetector.datasets.bsn_dataset import VideoDataSet, ProposalDataSet


def test_tem():
    temporal_dim = 3
    tem = TEM(temporal_dim=temporal_dim, tem_feat_dim=400, tem_hidden_dim=200)
    x = torch.randn(1, 100, 400)
    gt_bbox = torch.Tensor([[10, 20], [30, 40], [50, 60], [100, 123]])
    gt_bbox = gt_bbox.unsqueeze(1)
    gt_bbox /= 150

    # test anchors
    # And yes, this is linspace.
    anchor_mins, anchor_maxs = tem.anchors_tmins, tem.anchors_tmaxs
    gap = 1 / temporal_dim
    starts = torch.linspace(0, 1, temporal_dim + 1)[:-1]
    ends = starts + gap * 1

    assert len(anchor_mins) == len(anchor_maxs) == len(starts) == len(ends)
    assert torch.allclose(starts, torch.Tensor(anchor_mins)), \
        f'anchor_mins: {anchor_mins}, starts: {starts}'
    assert torch.allclose(ends, torch.Tensor(anchor_maxs)), \
        f'anchor_maxs: {anchor_maxs}, ends: {ends}'

    # test generate_labels
    labels = tem.generate_labels(gt_bbox)
    print(labels)


def test_dataset():
    ds = VideoDataSet(subset='train', temporal_scale = 100)
    index = 10
    video_data, anchor_xmin, anchor_xmax = ds._get_base_data(index)
    match_score_action, match_score_start, match_score_end, gt_bbox = ds._get_train_label(
        index, anchor_xmin, anchor_xmax)
    print(match_score_action[:3], match_score_action[-3:])
    print(gt_bbox)
    print(anchor_xmin)


if __name__ == '__main__':
    # test_tem()
    test_dataset()