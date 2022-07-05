from workoutdetector.settings import PROJ_ROOT, REPCOUNT_ANNO_PATH
import torch
from torchvision.io import read_image
import torchvision.transforms.functional as TF
import os
import os.path as osp
from os.path import join as osj
from fvcore.common.config import CfgNode
from workoutdetector.trainer import DataModule
from tempfile import TemporaryDirectory
from workoutdetector.trainer import train, DataModule


def _check_data(loader: torch.utils.data.DataLoader, num_class: int = 12):
    """Check data exists and annotation files are correct."""
    for x, y in loader:
        assert x.shape[-3:] == (3, 224, 224), f"{x.shape} is not (3, 224, 224)"
        assert 0 <= all(y) < num_class, f"{y} is not in range [0, {num_class})"


def test_DataModule():
    config = 'workoutdetector/configs/repcount_12_tsm.yaml'
    cfg = CfgNode(new_allowed=True)
    cfg.merge_from_file(config)
    cfg.trainer.fast_dev_run = True
    cfg.trainer.devices = 1
    cfg.log.wandb.offline = True
    cfg.data.data_root = osp.expanduser('~/data')
    cfg.data.train.anno = osp.expanduser('~/data/Binary/all-train.txt')
    cfg.data.val.anno = osp.expanduser('~/data/Binary/all-val.txt')
    cfg.data.test.anno = osp.expanduser('~/data/Binary/all-test.txt')

    assert osp.exists(cfg.data.data_root), f"{cfg.data.data_root} does not exist"
    assert osp.exists(cfg.data.train.anno), f"{cfg.data.train.anno} does not exist"
    assert osp.exists(cfg.data.val.anno), f"{cfg.data.val.anno} does not exist"
    assert osp.exists(cfg.data.test.anno), f"{cfg.data.test.anno} does not exist"

    num_class = cfg.model.num_class

    with TemporaryDirectory() as tmpdir:
        cfg.trainer.defaut_root_dir = tmpdir
        cfg.log.output_dir = osp.join(tmpdir, 'logs')
        datamodule = DataModule(cfg.data, is_train=True, num_class=num_class)
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        _check_data(train_loader)
        _check_data(val_loader)


def test_config():
    config = 'workoutdetector/configs/repcount_12_tsm.yaml'
    cfg = CfgNode(new_allowed=True)
    cfg.merge_from_file(config)
    cfg.trainer.fast_dev_run = True
    cfg.trainer.devices = 1
    cfg.log.wandb.offline = True
    with TemporaryDirectory() as tmpdir:
        cfg.trainer.defaut_root_dir = tmpdir
        cfg.log.output_dir = osp.join(tmpdir, 'logs')
        train(cfg)
