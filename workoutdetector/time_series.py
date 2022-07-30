import json
import os
import time
from typing import Any, Callable, List, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from fvcore.common.config import CfgNode
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from torch import Tensor, nn
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader, Dataset

from workoutdetector.datasets import FeatureDataset, RepcountHelper
from workoutdetector.utils import pred_to_count, analyze_count


class LSTMNet(nn.Module):

    def __init__(self, input_dim, num_layers, hidden_dim, output_dim, dropout, device):
        super(LSTMNet, self).__init__()
        self.rnn = nn.LSTM(input_dim,
                           hidden_dim,
                           num_layers,
                           batch_first=True,
                           dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                  nn.Linear(hidden_dim, output_dim))
        self.hidden_size = hidden_dim
        self.num_layers = num_layers
        self.device = device

    def forward(self, x):
        h = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to('cuda')
        c = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to('cuda')
        x, (h, c) = self.rnn(x, (h.detach(), c.detach()))
        o = self.head(h[-1])
        return o


class ConvNet(nn.Module):

    def __init__(self, input_size, num_layers, hidden_size, output_size):
        super(ConvNet, self).__init__()
        self.layers = nn.Sequential()


class LitModel(LightningModule):

    def __init__(self, cfg: CfgNode):
        super().__init__()
        if cfg.model.checkpoint is not None:
            print(f'Loading Lightning model from {cfg.model.checkpoint}')
            ckpt = self.load_from_checkpoint(cfg.model.checkpoint)
            cfg.merge_from_other_cfg(ckpt.cfg)
        else:
            self.save_hyperparameters()
        self.net = LSTMNet(cfg.model.input_dim, cfg.model.num_layers,
                           cfg.model.hidden_dim, cfg.model.output_dim, cfg.model.dropout,
                           self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.cfg = cfg
        self.example_input_array = torch.randn(cfg.model.example_input_array)

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        optimizer = Adam(self.net.parameters(), lr=self.cfg.optimizer.lr)
        scheduler = lr_scheduler.StepLR(optimizer,
                                        step_size=self.cfg.lr_scheduler.step,
                                        gamma=self.cfg.lr_scheduler.gamma)
        return [optimizer], [scheduler]

    def _trainval(self, batch, step):
        x, y = batch
        pred = self.net(x)
        loss = self.loss_fn(pred, y)
        acc = (pred.argmax(dim=1) == y).float().mean()
        self.log(f'{step}/loss', loss, sync_dist=True)
        self.log(f'{step}/acc',
                 acc,
                 prog_bar=True,
                 sync_dist=True,
                 on_epoch=True,
                 on_step=False)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._trainval(batch, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._trainval(batch, 'val')
        return loss

    def test_step(self, item, batch_idx):
        x, y = item['x'], item['y']
        window, stride = self.cfg.data.window, 1
        batch_x, batch_y = [], []
        x = x.squeeze(0)  # why one more dimension?
        for i in range(0, len(x), stride):
            if i + window > len(x):
                break
                # batch_x.append(x[i:])
                # batch_y.append(y[i:])
            else:
                batch_x.append(x[i:i + window])
                batch_y.append(y[i + window - 1])
        if not batch_x:
            batch_x = x.to(self.device).unsqueeze(0)
            batch_y = torch.tensor(y).to(self.device)
        else:
            batch_x = torch.stack(batch_x).to(self.device)
            batch_y = torch.tensor(batch_y).to(self.device)
        pred = self.net(batch_x).argmax(dim=1)
        acc = (pred == batch_y).float().mean()
        item.update({'pred': pred, 'acc': acc})
        return item

    def test_epoch_end(self, outputs):
        """Evaluate OBO and MAE and save to csv file."""

        df = pd.DataFrame(columns=[
            'name', 'action', 'split', 'acc', 'count', 'gt_count', 'reps', 'gt_reps'
        ])
        obo, mae = 0, 0
        for o in outputs:
            # minus 1 because 0 class means no action
            count, reps = pred_to_count(o['pred'] - 1, stride=1, step=1)
            diff = abs(count - o['gt_count'].item())
            obo += 1 if diff <= 1 else 0
            mae += diff
            df.loc[len(df)] = dict(name=o['video_name'][0],
                                   action=o['action'][0],
                                   split=o['split'][0],
                                   acc=o['acc'].item(),
                                   pred_count=count,
                                   gt_count=o['gt_count'].item(),
                                   pred_reps=reps,
                                   diff=diff)
        df.set_index('name', inplace=True)
        if self.cfg.model.checkpoint is not None:
            csv_path = os.path.join(self.trainer.default_root_dir,
                                    self.cfg.model.checkpoint)
        else:
            csv_path = f'{self.loggers[0].log_dir}/test_metrics.csv'
            self.log_dict({'OBO': obo / len(outputs), 'MAE': mae / len(outputs)})
        df.to_csv(csv_path)
        return obo, mae

    def predict_step(self, x):
        pred = self.net(x)
        return pred.argmax(dim=1)


def reps_to_label(reps, total, class_idx):
    y = [0] * total
    for start, end in zip(reps[::2], reps[1::2]):
        mid = (start + end) // 2
        y[start:mid] = [class_idx * 2 + 1] * (mid - start)
        y[mid:end] = [class_idx * 2 + 2] * (end - mid)  # plus 1 because no-class is 0
    return y


class FeatureDataModule(LightningDataModule):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def _dataloader(self, split, action, window, stride):
        return FeatureDataset(self.cfg.data.json_dir,
                              self.cfg.data.anno_path,
                              split=split,
                              action=action,
                              window=window,
                              stride=stride)

    def train_dataloader(self):
        train_ds = self._dataloader('train', 'all', self.cfg.data.window,
                                    self.cfg.data.stride)
        return DataLoader(train_ds, batch_size=self.cfg.data.batch_size, shuffle=True)

    def val_dataloader(self):
        val_ds = self._dataloader('val', 'all', self.cfg.data.window,
                                  self.cfg.data.stride)
        return DataLoader(val_ds, batch_size=self.cfg.data.batch_size, shuffle=False)

    def test_dataloader(self):
        test_ds = TestDataset(self.cfg)
        return DataLoader(test_ds, batch_size=1, shuffle=False)


class TestDataset(Dataset):

    def __init__(self, cfg: CfgNode):
        helper = RepcountHelper('', cfg.data.anno_path)
        test_data = list(helper.get_rep_data(['train', 'val', 'test'], ['all']).values())
        total_obo, total_err, total_acc = 0, 0, 0
        self.data: List[dict] = []
        for item in test_data:
            test_x: List[List[float]] = []
            js = json.load(
                open(
                    os.path.join(cfg.data.json_dir,
                                 cfg.data.template.format(item.video_name, 1, 1))))
            for i, v in js['scores'].items():
                test_x.append(list(v.values()))
            test_x = torch.tensor(test_x)  # type: ignore
            test_y = reps_to_label(item.reps, len(test_x),
                                   helper.classes.index(item.class_))
            assert len(test_x.shape) == 2, test_x.shape  # type: ignore
            self.data.append({
                'x': test_x,
                'y': test_y,
                'gt_count': item.count,
                'gt_reps': item.reps,
                'split': item.split,
                'action': item.class_,
                'video_name': item.video_name
            })

    def __getitem__(self, idx) -> dict:
        return self.data[idx]

    def __len__(self):
        return len(self.data)


def load_config():
    cfg = CfgNode(
        init_dict={
            'trainer': {
                'default_root_dir': 'exp/time_series',
                'max_epochs': 20,
                'devices': 1,
                'gpus': 1,
                'deterministic': True,
            },
            'data': {
                # dir of extracted features
                'json_dir': 'out/acc_0.841_epoch_26_20220711-191616_1x1',
                'template': '{}.stride_{}_step_{}.json',
                'anno_path': os.path.expanduser('~/data/RepCount/annotation.csv'),
                'batch_size': 32 * 1,
                'window': 100,
                'stride': 4,
            },
            'model': {
                'input_dim': 12,
                'output_dim': 13,
                'hidden_dim': 512,
                'num_layers': 3,  # LSTM
                'dropout': 0.5,
                'example_input_array': [1, 100, 12],  # batch, window, feature_dim
                'checkpoint': 'exp/time_series/acc_0.598_epoch_000-v1.ckpt',
            },
            'optimizer': {
                'lr': 1e-4,
            },
            'lr_scheduler': {
                'step': 10,
                'gamma': 0.1
            },
            'callbacks': {
                'modelcheckpoint': {
                    'save_top_k': 1,
                    'save_weights_only': False,
                    'monitor': 'val/acc',
                    'mode': 'max',
                    'dirpath': None,
                },
                'early_stopping': {
                    'enable': False,
                    'patience': 10,
                }
            },
            'log': {
                'log_every_n_steps': 20,
                'csv': {
                    'enable': True
                },
                'tensorboard': {
                    'enable': True
                },
                'wandb': {
                    'enable': False,
                    'offline': False,
                    'project': 'time_series'
                },
            },
            'seed': 42,
            'timestamp': time.strftime('%Y%m%d-%H%M%S'),
        })
    return cfg


def setup_module(cfg):
    pl.seed_everything(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = LitModel(cfg)
    data_module = FeatureDataModule(cfg)
    return model, data_module


def setup_logger(cfg: CfgNode):
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
        # wandb_logger.watch(model, log="all")
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


def setup_callbacks(model, log_dir, cfg: CfgNode):
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
        filename="val-acc={val/acc:.3f}-epoch={epoch:03d}" + f"-{cfg.timestamp}",
        auto_insert_metric_name=False)
    callbacks.append(checkpoint_callback)

    # EarlyStopping callback
    if cfg.callbacks.early_stopping.enable:
        early_stopping = EarlyStopping(monitor='train/loss',
                                       mode='min',
                                       patience=cfg.callbacks.early_stopping.patience)
        callbacks.append(early_stopping)
    return callbacks


def train():
    cfg = load_config()
    cfg.model.checkpoint = None
    model, data_module = setup_module(cfg)
    log_dir = os.path.join(cfg.trainer.default_root_dir, cfg.timestamp)

    logger = setup_logger(cfg)
    callbacks = setup_callbacks(model, log_dir, cfg)

    trainer = Trainer(
        **cfg.trainer,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=cfg.log.log_every_n_steps,
        # strategy=DDPStrategy(find_unused_parameters=True, process_group_backend='gloo'),
    )
    trainer.fit(model, data_module)


def test():
    cfg = load_config()
    model, data_module = setup_module(cfg)
    trainer = Trainer(devices=1, gpus=1)
    trainer.test(model, data_module)


if __name__ == '__main__':
    # train()
    test()
    # analyze_count(f'exp/time_series/test_metrics.csv')