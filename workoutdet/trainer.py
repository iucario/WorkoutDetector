import argparse
import os
import time
from os.path import join as osj
from typing import Any, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.transforms as T
import yaml
from fvcore.common.config import CfgNode
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from torch import Tensor
from torch import nn as nn
from torch.utils.data import DataLoader, Dataset

from workoutdet.data import FrameDataset
from workoutdet.optimizer import build_optim
from workoutdet.tsm import create_model


class LitModel(LightningModule):
    """Video classification model."""

    def __init__(self, cfg: CfgNode):
        super().__init__()
        self.save_hyperparameters()
        self.model = create_model(**cfg.model)
        if cfg.model.get('example_input_array') is not None:
            self.example_input_array = torch.randn(
                cfg.model.example_input_array)
        self.loss_module = nn.CrossEntropyLoss()
        self.cfg = cfg
        self.best_val_acc = 0.0

    def forward(self, x):
        x = x.reshape(-1, 3, 224, 224)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_module(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("train/acc",
                 acc,
                 prog_bar=True,
                 on_step=False,
                 on_epoch=True,
                 sync_dist=True)
        self.log('train/loss', loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Returns y_hat, y, step_acc for calculating best val_acc per epoch."""

        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_module(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("val/acc",
                 acc,
                 prog_bar=True,
                 on_step=False,
                 on_epoch=True,
                 sync_dist=True)
        self.log('val/loss', loss, sync_dist=True)
        correct = (y_hat.argmax(dim=1) == y).sum().item()
        total = len(y)
        return {'correct': correct, 'total': total}

    def validation_epoch_end(self, outputs):
        """Calculate and log best val_acc per epoch.

        Example::
            
            world_size = 8, batch_size = 2
            >>> return {'correct': correct, 'total': total}

            ===> gathered: 
                [{'correct': tensor([0, 0, 0, 0, 0, 0, 1, 0], device='cuda:2', dtype=torch.int32),
                'total': tensor([4, 4, 4, 4, 4, 4, 4, 4], device='cuda:2', dtype=torch.int32)},
                {'correct': tensor([1, 0, 1, 0, 1, 0, 0, 0], device='cuda:2', dtype=torch.int32),
                'total': tensor([4, 4, 4, 4, 4, 4, 4, 4], device='cuda:2', dtype=torch.int32)}]
        """
        gathered = self.all_gather(outputs)  # shape: (world_size, batch, ...)
        correct = sum([x['correct'].sum().item() for x in gathered])
        total = sum([x['total'].sum().item() for x in gathered])
        acc = correct / total
        if self.trainer.is_global_zero:
            self.best_val_acc = max(self.best_val_acc, acc)
            self.log('val/best_acc', self.best_val_acc, rank_zero_only=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_module(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("test/acc", acc, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test/loss', loss, sync_dist=True)

    def predict_step(self, batch, batch_idx):
        return self(batch)

    def configure_optimizers(self):
        n_iter_per_epoch = self.trainer.estimated_stepping_batches
        optimizer, scheduler = build_optim(self.cfg,
                                           self.model,
                                           n_iter_per_epoch=n_iter_per_epoch)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val/loss",
        }


class DataModule(LightningDataModule):
    """Frame dataset

    Args:
        cfg (CfgNode): configs of cfg.data
        is_train: bool, train or test. Default True
    """

    def __init__(self,
                 cfg: CfgNode,
                 is_train: bool = True,
                 num_class: int = 2) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_class = num_class
        # self._check_data()
        self.train_transform = T.Compose([
            T.ConvertImageDtype(torch.float32),
            T.Resize(256),
            T.RandomCrop((224, 224)),
            T.RandomHorizontalFlip(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.test_transform = T.Compose([
            T.ConvertImageDtype(torch.float32),
            T.Resize(256),
            T.CenterCrop((224, 224)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _check_data(self):
        """Check data exists and annotation files are correct."""

        print(f"Checking {self.cfg.dataset_type} at {self.cfg.data_root}")
        for split in ['train', 'val', 'test']:
            ds = self.build_dataset(self.cfg, split)
            assert len(ds) > 0, f"{split} dataset is empty"
            assert ds[0]
        print("Data check passed.")

    def build_loader(self, anno_path: str, prefix: str,
                     is_test: bool) -> DataLoader:
        dataset = FrameDataset(
            data_root=self.cfg.data_root,
            anno_path=anno_path,
            data_prefix=prefix,
            num_segments=self.cfg.num_segments,
            filename_tmpl=self.cfg.filename_tmpl,
            transform=self.train_transform
            if not is_test else self.test_transform,
            anno_col=self.cfg.anno_col,
            is_test=is_test,
        )
        return DataLoader(
            dataset,
            num_workers=self.cfg.num_workers,
            batch_size=self.cfg.batch_size,
            shuffle=True if not is_test else False,
        )

    def train_dataloader(self):
        return self.build_loader(self.cfg.train.anno,
                                 self.cfg.train.data_prefix,
                                 is_test=False)

    def val_dataloader(self):
        return self.build_loader(self.cfg.val.anno,
                                 self.cfg.val.data_prefix,
                                 is_test=False)

    def test_dataloader(self):
        if self.cfg.test.anno:
            return self.build_loader(self.cfg.test.anno,
                                     self.cfg.test.data_prefix,
                                     is_test=True)
        else:
            return self.val_dataloader()


def setup_module(cfg: CfgNode) -> LightningModule:
    """Load checkpoint if any and setup model."""
    if cfg.model.checkpoint is not None:
        print(f'Loading Lightning model from {cfg.model.checkpoint}')
        ckpt = LitModel.load_from_checkpoint(cfg.model.checkpoint)
        cfg.merge_from_other_cfg(ckpt.cfg)

    model = LitModel(cfg)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return model


def setup_logger(cfg: CfgNode) -> list:
    """Return a list of Lightning loggers."""
    log_dir = os.path.join(cfg.trainer.default_root_dir, cfg.timestamp)
    cfg_dict = cfg_dict = yaml.safe_load(cfg.dump())
    logger: List[Any] = []
    if cfg.log.wandb.enable:
        wandb_logger = WandbLogger(
            save_dir=log_dir,
            project=cfg.log.wandb.project,
            name=cfg.log.wandb.name,
            offline=cfg.log.wandb.offline,
        )
        wandb_logger.log_hyperparams(cfg_dict)
        logger.append(wandb_logger)

    if cfg.log.tensorboard.enable:
        tensorboard_logger = TensorBoardLogger(save_dir=log_dir,
                                               name='tensorboard',
                                               default_hp_metric=False)
        tensorboard_logger.log_hyperparams(cfg_dict)
        logger.append(tensorboard_logger)

    if cfg.log.csv.enable:
        csv_logger = CSVLogger(save_dir=log_dir, name='csv')
        csv_logger.log_hyperparams(cfg_dict)
        logger.append(csv_logger)
    return logger


def setup_callbacks(model, log_dir, cfg: CfgNode) -> list:
    """Return a list of Lightning callbacks."""
    callbacks: List[Any] = []

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(log_momentum=True)
    callbacks.append(lr_monitor)

    # ModelCheckpoint callback
    if model.global_rank == 0 and not os.path.isdir(log_dir):
        print(f'Create checkpoint directory: {log_dir}')
        os.makedirs(log_dir)
    cfg.callbacks.modelcheckpoint.dirpath = log_dir
    checkpoint_callback = ModelCheckpoint(
        **cfg.callbacks.modelcheckpoint,
        filename="val-acc={val/acc:.3f}-epoch={epoch:03d}" +
        f"-{cfg.timestamp}",
        auto_insert_metric_name=False)
    callbacks.append(checkpoint_callback)

    # EarlyStopping callback
    if cfg.callbacks.early_stopping.enable:
        early_stopping = EarlyStopping(
            monitor='train/loss',
            mode='min',
            patience=cfg.callbacks.early_stopping.patience)
        callbacks.append(early_stopping)
    return callbacks


def train(cfg: CfgNode) -> None:
    cfg.model.checkpoint = None
    data_module = DataModule(cfg.data)
    model = setup_module(cfg)

    log_dir = os.path.join(cfg.trainer.default_root_dir, cfg.timestamp)

    logger = setup_logger(cfg)
    callbacks = setup_callbacks(model, log_dir, cfg)

    trainer = Trainer(
        **cfg.trainer,
        logger=logger,
        callbacks=callbacks,
        # strategy=DDPStrategy(find_unused_parameters=True, process_group_backend='gloo'),
    )
    trainer.fit(model, data_module)


def test(cfg: CfgNode) -> None:
    data_module = DataModule(cfg.data)
    model = setup_module(cfg)
    trainer = Trainer(default_root_dir=cfg.trainer.default_root_dir,
                      devices=1,
                      gpus=1)
    trainer.test(model, data_module)


def export_model(ckpt: str, out_path: Optional[str] = None) -> None:
    """Export model to torchscript."""
    model = setup_module(cfg)
    model.eval()
    if out_path is None:
        out_path = ckpt.replace('.ckpt', '.pt')
    model.to_torchscript(out_path,
                         input_sample=model.example_input_array,
                         export_params=True)


def cfg_to_dict(cfg: CfgNode) -> dict:
    x = cfg.dump()
    y = yaml.safe_load(x)
    return y


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='PyTorch Lightning Trainer')
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default=osj("workoutdet/repcount.yaml"),
        type=str,
    )
    parser.add_argument(
        "opts",
        help="See workoutdet/repcount.yaml for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args(argv)


def load_config(args) -> CfgNode:
    """Given the arguemnts, load and initialize the configs.

    Args:
        args (argument): arguments includes `cfg_file`.
    """
    cfg = CfgNode(yaml.safe_load(open('workoutdet/repcount.yaml')))
    cfg.merge_from_file(args.cfg_file)
    if args.opts is not None:
        cfg.merge_from_list(args.opts)
    return cfg


def main(cfg: CfgNode) -> None:
    assert cfg.model.num_frames == cfg.data.num_frames, \
        f"model.num_frames ({cfg.model.num_frames}) != data.num_frames ({cfg.data.num_frames})"
    pl.seed_everything(cfg.seed)
    timestamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    if not cfg.timestamp:
        cfg.timestamp = timestamp
    if cfg.train:
        train(cfg)
    else:
        test(cfg)


if __name__ == '__main__':

    args = parse_args()
    cfg = load_config(args)
    main(cfg)
