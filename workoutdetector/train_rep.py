import argparse
import copy
import os
import os.path as osp
import time
import warnings
from os.path import join as osj
from typing import List

import mmcv
import torch
import torch.distributed as dist
from mmaction.apis import init_random_seed, train_model
from mmaction.datasets import build_dataset
from mmaction.datasets.base import BaseDataset
from mmaction.datasets.builder import DATASETS
from mmaction.models import build_model
from mmaction.utils import (collect_env, get_root_logger, register_module_hooks,
                            setup_multi_processes)
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist, set_random_seed
from mmcv.utils import get_git_hash

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
    if len(cfg.gpu_ids) > 1:
        distributed = True
        init_dist('pytorch', **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)
    else:
        distributed = False
    print("==> Distributed:", distributed)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    cfg.work_dir = osp.join(cfg.work_dir, timestamp)
    mmcv.mkdir_or_exist(cfg.work_dir)
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    meta['env_info'] = env_info
    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config: {cfg.pretty_text}')

    # set random seeds
    seed = init_random_seed(cfg.seed, distributed=distributed)
    logger.info(f'Set random seed to {seed}, deterministic: True')
    set_random_seed(seed, deterministic=True)
    cfg.setdefault('module_hooks', [])
    meta['seed'] = seed
    meta['config_name'] = osp.basename(args.cfg)
    meta['work_dir'] = osp.basename(cfg.work_dir.rstrip('/\\'))

    # Build the dataset
    datasets = [build_dataset(cfg.data.train)]

    # Build the recognizer
    model = build_model(cfg.model,
                        train_cfg=cfg.model.train_cfg,
                        test_cfg=cfg.model.test_cfg)
    print('==> train')
    train_model(model,
                datasets,
                cfg,
                distributed=distributed,
                validate=True,
                test=dict(test_last=False, test_best=True),
                timestamp=timestamp,
                meta=meta)


def main(args):
    cfg = Config.fromfile(args.cfg)
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    setup_multi_processes(cfg)
    cfg.seed = 0
    num_gpus = torch.cuda.device_count()
    cfg.gpu_ids = range(num_gpus)

    if args.action == 'all':
        cfg.model.cls_head.num_classes = 12
    else:
        cfg.model.cls_head.num_classes = 2

    assert osp.exists(cfg.data.train.ann_file), f'{cfg.data.train.ann_file} not found'
    assert osp.exists(cfg.data.val.ann_file), f'{cfg.data.val.ann_file} not found'
    assert osp.exists(cfg.data.test.ann_file), f'{cfg.data.test.ann_file} not found'

    # cfg.resume_from = osp.join(cfg.work_dir, 'latest.pth')

    cfg.log_config = dict(interval=10,
                          hooks=[
                              dict(type='TextLoggerHook'),
                              dict(type='TensorboardLoggerHook'),
                              dict(type='WandbLoggerHook',
                                   init_kwargs=dict(project='mmaction-rep-12',
                                                    config={**cfg}))
                          ])

    if args.export:  # TODO: deal with sync batchnorm
        input_sample = torch.randn(1, 8, 3, 224, 224)
        input_sample = input_sample.cuda()
        output = 'mmaction_model.onnx'
        model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
        model.cuda().eval()
        if hasattr(model, 'forward_dummy'):
            from functools import partial
            model.forward = partial(model.forward_dummy, softmax=False)
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
    CFG = 'workoutdetector/configs/tsm_MultiActionRepCount_sthv2.py'
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', default=CFG, type=str, help='config file path')
    parser.add_argument('--export', action='store_true', help='export model')
    parser.add_argument('--ckpt', type=str, help='checkpoint to load')
    parser.add_argument('-a', '--action', type=str, default='all', choices=ACTIONS)
    parser.add_argument("--local_rank", type=int, default=0)

    args_export_pull_up = [
        '--action=pull_up', '--data-prefix=data/RepCount/rawframes/',
        '--ann-dir=data/relabeled/pull_up', '--export',
        '--ckpt=work_dirs/tsm_MultiActionRepCount_sthv2_20220625-224626/best_top1_acc_epoch_5.pth'
    ]
    args_timesformer = [
        '--action=all',
        '--cfg=workoutdetector/configs/timesformer_div_8x4x1_k400.py',
    ]
    args = parser.parse_args()

    main(args)
