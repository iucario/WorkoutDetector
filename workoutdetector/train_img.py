import argparse
import os
import os.path as osp
import time
from os.path import join as osj
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
import timm
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import yaml
from fvcore.common.config import CfgNode
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import (LearningRateMonitor, ModelCheckpoint,
                                         early_stopping)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from workoutdetector.datasets import build_dataset
from workoutdetector.settings import PROJ_ROOT


class LitModel(LightningModule):

    def __init__(self, cfg: CfgNode):
        super().__init__()
        self.example_input_array = torch.randn(1, 3, 224, 224)
        self.save_hyperparameters()
        backbone = timm.create_model(cfg.model.backbone_model,
                                     pretrained=True,
                                     num_classes=cfg.model.num_class)
        self.classifier = backbone
        self.loss_module = nn.CrossEntropyLoss()
        self.cfg = cfg
        self.best_val_acc = 0

    def forward(self, x):
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_module(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("train/acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_module(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("val/acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val/loss', loss)
        return y_hat.argmax(dim=1) == y

    def validation_epoch_end(self, batch_parts_outputs):
        outputs = self.all_gather(batch_parts_outputs)
        y_hat = torch.cat([output['y_hat'] for output in outputs], dim=0)
        y = torch.cat([output['y'] for output in outputs], dim=0)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.best_val_acc = max(self.best_val_acc, acc.item())
        if self.trainer.is_global_zero:
            self.log('val/best_acc', acc, rank_zero_only=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_module(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("test/acc", acc, on_step=False, on_epoch=True)
        self.log('test/loss', loss, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def configure_optimizers(self):
        cfg = self.cfg
        OPTIMIZER = cfg.optimizer.method.lower()
        SCHEDULER = cfg.lr_scheduler.policy.lower()
        if OPTIMIZER == 'sgd':
            optimizer = optim.SGD(self.parameters(),
                                  lr=cfg.optimizer.lr,
                                  momentum=cfg.optimizer.momentum,
                                  weight_decay=cfg.optimizer.weight_decay)
        elif OPTIMIZER == 'adamw':
            optimizer = optim.AdamW(self.parameters(),
                                    lr=cfg.optimizer.lr,
                                    eps=cfg.optimizer.eps,
                                    weight_decay=cfg.optimizer.weight_decay)
        else:
            raise NotImplementedError(
                f'Not implemented optimizer: {cfg.optimizer.method}')
        if SCHEDULER == 'steplr':
            scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                  step_size=cfg.lr_scheduler.step,
                                                  gamma=cfg.lr_scheduler.gamma)
        else:
            raise NotImplementedError(f'Not implemented lr schedular: {cfg.lr_schedular}')
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val/loss",
        }


class DataModule(LightningDataModule):
    """General image dataset
    label text files of [image.png class] are required

    Args:
        cfg (CfgNode): configs of cfg.data
        is_train: bool, train or test. Default True
    """

    def __init__(self, cfg: CfgNode, is_train: bool = True, num_class: int = 0) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_class = num_class
        self._check_data()

    def _check_data(self):
        """Check data exists and annotation files are correct."""
        for split in ['train', 'val', 'test']:
            ds = build_dataset(self.cfg.dataset_type, self.cfg, split)
            for i, (x, y) in enumerate(ds):
                assert type(x) == torch.Tensor, f"{type(x) is not Tensor}"
                assert 0 <= y < self.num_class, f"{y} is not in [0, {self.num_class})"

    def train_dataloader(self):
        train_set = build_dataset(self.cfg.dataset_type, self.cfg, 'train')
        loader = DataLoader(train_set,
                            num_workers=self.cfg.num_workers,
                            batch_size=self.cfg.batch_size,
                            shuffle=True)
        return loader

    def val_dataloader(self):
        val_set = build_dataset(self.cfg.dataset_type, self.cfg, 'val')
        loader = DataLoader(val_set,
                            num_workers=self.cfg.num_workers,
                            batch_size=self.cfg.batch_size,
                            shuffle=False)
        return loader

    def test_dataloader(self):
        if self.cfg.test.anno:
            test_set = build_dataset(self.cfg.dataset_type, self.cfg, 'test')
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
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
    )
    trainer.test(model, data_module)


def train(cfg: CfgNode) -> None:
    data_module = DataModule(cfg.data, num_class=cfg.model.num_class)
    model = LitModel(cfg)

    timenow = time.strftime('%Y%m%d-%H%M%S', time.localtime())

    # callbacks
    CALLBACKS: List[Any] = []

    lr_monitor = LearningRateMonitor(logging_interval='step')
    CALLBACKS.append(lr_monitor)

    # ModelCheckpoint callback
    if cfg.callbacks.modelcheckpoint.dirpath:
        DIRPATH = cfg.callbacks.modelcheckpoint.dirpath
    else:
        DIRPATH = osj(cfg.trainer.default_root_dir, 'checkpoints')
    checkpoint_callback = ModelCheckpoint(
        save_top_k=cfg.callbacks.modelcheckpoint.save_top_k,
        save_weights_only=cfg.callbacks.modelcheckpoint.save_weights_only,
        monitor=cfg.callbacks.modelcheckpoint.monitor,
        mode=cfg.callbacks.modelcheckpoint.mode,
        dirpath=DIRPATH,
        filename="best-val-acc={val/acc:.2f}-epoch={epoch:02d}" + f"-{timenow}",
        auto_insert_metric_name=False)
    CALLBACKS.append(checkpoint_callback)

    if cfg.trainer.early_stopping:
        early_stop = early_stopping.EarlyStopping(monitor='train/loss',
                                                  mode='min',
                                                  patience=cfg.trainer.patience)
        CALLBACKS.append(early_stop)

    # loggers
    cfg_dict = cfg_to_dict(cfg)
    LOGGER: List[Any] = []
    if cfg.log.wandb.enable:
        wandb_logger = WandbLogger(
            save_dir=osj(cfg.log.output_dir),
            project=cfg.log.wandb.project,
            name=cfg.log.name,
            offline=cfg.log.wandb.offline,
        )
        wandb_logger.log_hyperparams(cfg_dict)
        wandb_logger.watch(model, log="all")
        LOGGER.append(wandb_logger)

    if cfg.log.tensorboard.enable:
        tensorboard_logger = TensorBoardLogger(save_dir=cfg.log.output_dir,
                                               name=cfg.log.name,
                                               default_hp_metric=False)
        tensorboard_logger.log_hyperparams(cfg_dict,
                                           metrics={
                                               'train/acc': 0,
                                               'val/acc': 0,
                                               'test/acc': 0,
                                               'train/loss': -1,
                                               'val/loss': -1,
                                               'test/loss': -1
                                           })
        LOGGER.append(tensorboard_logger)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    trainer = Trainer(
        default_root_dir=cfg.trainer.default_root_dir,
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        logger=LOGGER,
        callbacks=CALLBACKS,
        auto_lr_find=cfg.trainer.auto_lr_find,
        log_every_n_steps=cfg.log.log_every_n_steps,
        fast_dev_run=cfg.trainer.fast_dev_run,
        strategy=DDPStrategy(find_unused_parameters=False),
    )

    trainer.fit(model, data_module)

    model.load_from_checkpoint(checkpoint_callback.best_model_path)
    print(f"===>Best model saved at:\n{checkpoint_callback.best_model_path}")

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
        default=osj(PROJ_ROOT, "workoutdetector/configs/lit_img.yaml"),
        type=str,
    )
    parser.add_argument(
        "opts",
        help="See workoutdetector/configs/lit_img.yaml for all options",
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
    cfg = CfgNode(new_allowed=True)
    cfg.merge_from_file(args.cfg_file)
    if args.opts is not None:
        cfg.merge_from_list(args.opts)
    return cfg


def main(cfg: CfgNode) -> None:
    pl.seed_everything(cfg.seed)

    if cfg.train:
        train(cfg)
    else:
        test(cfg)


if __name__ == '__main__':

    args = parse_args()
    cfg = load_config(args)
    main(cfg)
