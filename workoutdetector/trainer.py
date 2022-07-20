import argparse
import os
import os.path as osp
import time
from os.path import join as osj
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import yaml
from fvcore.common.config import CfgNode
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import (LearningRateMonitor, ModelCheckpoint,
                                         early_stopping)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader

from workoutdetector.datasets import build_dataset
from workoutdetector.models import build_model, build_optim


class LitModel(LightningModule):
    """Video classification model."""

    def __init__(self, cfg: CfgNode):
        super().__init__()
        self.save_hyperparameters()
        self.model = build_model(cfg)
        if cfg.model.get('example_input_array'):
            self.example_input_array = torch.randn(cfg.model.example_input_array)
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

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
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

    def __init__(self, cfg: CfgNode, is_train: bool = True, num_class: int = 2) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_class = num_class
        # self._check_data()

    def _check_data(self):
        """Check data exists and annotation files are correct."""

        print(f"Checking {self.cfg.dataset_type} at {self.cfg.data_root}")
        for split in ['train', 'val', 'test']:
            ds = build_dataset(self.cfg, split)
            assert len(ds) > 0, f"{split} dataset is empty"
            assert ds[0]
        print("Data check passed.")

    def train_dataloader(self):
        train_set = build_dataset(self.cfg, 'train')
        loader = DataLoader(train_set,
                            num_workers=self.cfg.num_workers,
                            batch_size=self.cfg.batch_size,
                            shuffle=True)
        return loader

    def val_dataloader(self):
        val_set = build_dataset(self.cfg, 'val')
        loader = DataLoader(val_set,
                            num_workers=self.cfg.num_workers,
                            batch_size=self.cfg.batch_size,
                            shuffle=False)
        return loader

    def test_dataloader(self):
        if self.cfg.test.anno:
            test_set = build_dataset(self.cfg, 'test')
            loader = DataLoader(test_set,
                                num_workers=self.cfg.num_workers,
                                batch_size=self.cfg.batch_size,
                                shuffle=False)
            return loader
        else:
            return self.val_dataloader()


def test(cfg: CfgNode) -> None:
    data_module = DataModule(cfg.data, is_train=False)
    model = LitModel.load_from_checkpoint(cfg.checkpoint)
    trainer = Trainer(
        default_root_dir=cfg.trainer.default_root_dir,
        devices=1,
    )
    trainer.test(model, data_module)


def train(cfg: CfgNode) -> None:
    data_module = DataModule(cfg.data, num_class=cfg.model.num_class)
    data_module._check_data()
    model = LitModel(cfg)

    LOG_DIR = os.path.join(cfg.trainer.default_root_dir, cfg.timestamp)

    # ------------------------------------------------------------------- #
    # Callbacks
    # ------------------------------------------------------------------- #
    CALLBACKS: List[Any] = []

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch', log_momentum=True)
    CALLBACKS.append(lr_monitor)

    # ModelCheckpoint callback
    if cfg.callbacks.modelcheckpoint.dirpath:
        DIRPATH = cfg.callbacks.modelcheckpoint.dirpath
    else:
        DIRPATH = LOG_DIR
    if model.global_rank == 0 and not os.path.isdir(DIRPATH):
        print(f'Create checkpoint directory: {DIRPATH}')
        os.makedirs(DIRPATH)
    cfg.callbacks.modelcheckpoint.dirpath = DIRPATH
    checkpoint_callback = ModelCheckpoint(
        **cfg.callbacks.modelcheckpoint,
        filename="best-val-acc={val/acc:.3f}-epoch={epoch:02d}" + f"-{cfg.timestamp}",
        auto_insert_metric_name=False)
    CALLBACKS.append(checkpoint_callback)

    # EarlyStopping callback
    if cfg.callbacks.early_stopping.enable:
        early_stop = early_stopping.EarlyStopping(
            monitor='train/loss',
            mode='min',
            patience=cfg.callbacks.early_stopping.patience)
        CALLBACKS.append(early_stop)

    # ------------------------------------------------------------------- #
    # Loggers
    # ------------------------------------------------------------------- #
    cfg_dict = cfg_to_dict(cfg)
    LOGGER: List[Any] = []
    if cfg.log.wandb.enable:
        wandb_logger = WandbLogger(
            save_dir=LOG_DIR,
            project=cfg.log.wandb.project,
            name=cfg.log.wandb.name,
            offline=cfg.log.wandb.offline,
        )
        wandb_logger.log_hyperparams(cfg_dict)
        wandb_logger.watch(model, log="all")
        LOGGER.append(wandb_logger)

    if cfg.log.tensorboard.enable:
        tensorboard_logger = TensorBoardLogger(save_dir=LOG_DIR,
                                               name='tensorboard',
                                               default_hp_metric=False)
        tensorboard_logger.log_hyperparams(cfg_dict)
        LOGGER.append(tensorboard_logger)

    if cfg.log.csv.enable:
        csv_logger = CSVLogger(save_dir=LOG_DIR, name='csv')
        csv_logger.log_hyperparams(cfg_dict)
        csv_logger.log_graph(model)
        LOGGER.append(csv_logger)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ------------------------------------------------------------------- #
    # Trainer
    # ------------------------------------------------------------------- #
    trainer = Trainer(
        **cfg.trainer,
        logger=LOGGER,
        callbacks=CALLBACKS,
        log_every_n_steps=cfg.log.log_every_n_steps,
        strategy=DDPStrategy(find_unused_parameters=True, process_group_backend='gloo'),
    )

    trainer.fit(model, data_module)

    # ------------------------------------------------------------------- #
    # Test using best val acc model
    # ------------------------------------------------------------------- #
    if not cfg.trainer.fast_dev_run and model.global_rank == 0:
        model.load_from_checkpoint(checkpoint_callback.best_model_path)
        print(f"===>Best model saved at:\n{checkpoint_callback.best_model_path}")

    if model.global_rank == 0:
        trainer = Trainer(logger=LOGGER, callbacks=CALLBACKS, devices=1, gpus=1)
        trainer.test(model, data_module)


def export_model(ckpt: str, onnx_path: Optional[str] = None) -> None:
    model = LitModel.load_from_checkpoint(ckpt)
    model.eval()
    if onnx_path is None:
        onnx_path = ckpt.replace('.ckpt', '.onnx')
    model.to_onnx(onnx_path, input_sample=model.example_input_array, export_params=True)


def cfg_to_dict(cfg: CfgNode) -> dict:
    x = cfg.dump()
    y = yaml.safe_load(x)
    return y


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train image model')
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default=osj("workoutdetector/configs/repcount_12_tsm.yaml"),
        type=str,
    )
    parser.add_argument(
        "opts",
        help="See workoutdetector/configs/defaults.yaml for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args(argv)


def load_config(args) -> CfgNode:
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `cfg_file`.
    """
    cfg = CfgNode(yaml.safe_load(open('workoutdetector/configs/defaults.yaml')))
    cfg.merge_from_file(args.cfg_file)
    if args.opts is not None:
        cfg.merge_from_list(args.opts)
    return cfg


def main(cfg: CfgNode) -> None:
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
