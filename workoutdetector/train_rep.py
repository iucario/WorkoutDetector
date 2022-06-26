import argparse
import copy
import os
import os.path as osp
from os.path import join as osj
from typing import List
import torch
import mmcv
from mmaction.apis import train_model
from mmaction.datasets import build_dataset
from mmaction.datasets.base import BaseDataset
from mmaction.datasets.builder import DATASETS
from mmaction.models import build_model
from mmcv import Config
from mmcv.runner import set_random_seed

from workoutdetector.settings import PROJ_ROOT


@DATASETS.register_module()
class MultiActionRepCount(BaseDataset):
    """
    Note:
        label.txt has the following format:
            `dir/to/video/frames start_index total_frames label`
        
        Labels are built using `scripts/build_label_list.py`
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_prefix=None,
                 test_mode=False,
                 modality='RGB',
                 filename_tmpl='img_{:05}.jpg'):
        super(MultiActionRepCount, self).__init__(ann_file, pipeline, data_prefix,
                                                  test_mode, modality)

        self.filename_tmpl = filename_tmpl
        self.data_prefix = data_prefix

    def load_annotations(self) -> List[dict]:
        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                if line.startswith("directory"):  # what is this?
                    continue
                frame_dir, start_index, total_frames, label = line.split()
                if self.data_prefix is not None and int(total_frames) > 0:
                    frame_dir = os.path.join(self.data_prefix, frame_dir)
                video_infos.append(
                    dict(frame_dir=frame_dir,
                         start_index=int(start_index),
                         total_frames=int(total_frames),
                         label=int(label)))
        return video_infos

    def prepare_train_frames(self, idx):
        assert self.video_infos
        results = copy.deepcopy(self.video_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        assert self.video_infos
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
    mmcv.mkdir_or_exist(cfg.work_dir)
    train_model(model, datasets, cfg, distributed=False, validate=True)


def main(args):
    config = os.path.join(PROJ_ROOT,
                          'workoutdetector/configs/tsm_MultiActionRepCount_sthv2.py')
    cfg = Config.fromfile(config)
    cfg.seed = 0
    set_random_seed(0, deterministic=True)

    if args.action == 'all':
        cfg.model.cls_head.num_classes = (len(ACTIONS) - 1) * 2
    else:
        cfg.model.cls_head.num_classes = 2

    if args.ann_dir:
        ann_dir = args.ann_dir
    else:
        ann_dir = os.path.join(PROJ_ROOT, 'data/Binary')
    cfg.data.train.ann_file = os.path.join(ann_dir, f'train.txt')
    cfg.data.val.ann_file = os.path.join(ann_dir, f'val.txt')
    cfg.data.test.ann_file = os.path.join(ann_dir, f'test.txt')
    assert osp.exists(cfg.data.train.ann_file), f'{cfg.data.train.ann_file} not found'
    assert osp.exists(cfg.data.val.ann_file), f'{cfg.data.val.ann_file} not found'
    assert osp.exists(cfg.data.test.ann_file), f'{cfg.data.test.ann_file} not found'

    if args.data_prefix:
        cfg.data.train.data_prefix = args.data_prefix
        cfg.data.val.data_prefix = args.data_prefix
        cfg.data.test.data_prefix = args.data_prefix
        assert osp.isdir(args.data_prefix), f'{args.data_prefix} not found'

    # cfg.resume_from = osp.join(cfg.work_dir, 'latest.pth')

    cfg.log_config = dict(
        interval=10,
        hooks=[
            dict(type='TextLoggerHook'),
            dict(type='TensorboardLoggerHook'),
            #   dict(type='WandbLoggerHook',
            #        init_kwargs=dict(project='playground-tsm', config={**cfg}))
        ])
    print(cfg.pretty_text)

    if args.export:
        input_sample = torch.randn( 1,8,3, 224, 224)
        input_sample = input_sample.cuda()
        output = 'mmaction_model.onnx'
        model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
        model.cuda().eval()
        if hasattr(model, 'forward_dummy'):
            from functools import partial
            model.forward = partial(model.forward_dummy, softmax=False)
        elif hasattr(model, '_forward') and args.is_localizer:
            model.forward = model._forward
        else:
            raise NotImplementedError(
                'Please implement the forward method for exporting.')
        torch.onnx.export(model,
                      input_sample,
                      output,
                      export_params=True,
                      keep_initializers_as_inputs=True,
                      opset_version=11)
    else:
        train(cfg)


if __name__ == '__main__':
    ACTIONS = ['situp', 'push_up', 'pull_up', 'jump_jack', 'squat', 'front_raise', 'all']
    parser = argparse.ArgumentParser()
    parser.add_argument('--export', action='store_true', help='export model')
    parser.add_argument('--ckpt', type=str, help='checkpoint to load')
    parser.add_argument('-a', '--action', type=str, default='all', choices=ACTIONS)
    parser.add_argument('--data-prefix',
                        dest='data_prefix',
                        type=str,
                        default=None,
                        help='data prefix added to path in annotation file')
    parser.add_argument(
        '--ann-dir',
        dest='ann_dir',
        type=str,
        default=None,
        help='annotation directory. Expects train.txt, val.txt, test.txt in it')

    args_pull_up = [
        '--action=pull_up', '--data-prefix=data/RepCount/rawframes/',
        '--ann-dir=data/relabeled/pull_up', '--export',
        '--ckpt=work_dirs/tsm_MultiActionRepCount_sthv2_20220625-224626/best_top1_acc_epoch_5.pth'
    ]

    args = parser.parse_args(args_pull_up)

    main(args)
