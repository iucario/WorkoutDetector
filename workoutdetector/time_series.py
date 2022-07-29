import numpy as np
import json
import os
import pickle
from typing import Any, Callable, List, Tuple, Union

import pytorch_lightning as pl
import torch
import yaml
from fvcore.common.config import CfgNode
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import (LearningRateMonitor, ModelCheckpoint,
                                         early_stopping)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from scipy.stats import norm
from torch import Tensor, nn
from torch.autograd import Variable
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from workoutdetector.datasets import FeatureDataset, RepcountHelper
from workoutdetector.utils import pred_to_count


class LSTMNet(nn.Module):

    def __init__(self, input_dim, num_layers, hidden_dim, output_dim, device):
        super(LSTMNet, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                  nn.Linear(hidden_dim, output_dim))
        self.hidden_size = hidden_dim
        self.num_layers = num_layers
        self.device = device

    def forward(self, x):
        h = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to('cuda')
        c = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to('cuda')
        x, (h, c) = self.rnn(x, (h, c))
        o = self.head(h[-1])
        return o


class ConvNet(nn.Module):

    def __init__(self, input_size, num_layers, hidden_size, output_size):
        super(ConvNet, self).__init__()
        self.layers = nn.Sequential()


class LitModel(LightningModule):

    def __init__(self, cfg: CfgNode):
        super().__init__()
        cfg_dict = yaml.safe_load(cfg.dump())
        self.save_hyperparameters()
        if self.logger:
            self.logger.log_hyperparams(cfg_dict)
        self.net = LSTMNet(cfg.model.input_dim, cfg.model.num_layers,
                           cfg.model.hidden_dim, cfg.model.output_dim, self.device)
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

    def _dataloader(self, split, action, window, stride):
        return FeatureDataset(self.cfg.data.json_dir,
                              self.cfg.data.anno_path,
                              split=split,
                              action=action,
                              window=window,
                              stride=stride)

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
        x = x.squeeze(0) # why one more dimension?
        for i in range(0, len(x)-window+1, stride):
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
        return pred, acc

    def test_epoch_end(self, outputs):
        print(len(outputs))
        print(outputs[0])
        result = dict()
            

    def predict_step(self, x):
        pred = self.net(x)
        return pred.argmax(dim=1)

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


def reps_to_label(reps, total, class_idx):
    y = [0] * total
    for start, end in zip(reps[::2], reps[1::2]):
        mid = (start + end) // 2
        y[start:mid] = [class_idx * 2 + 1] * (mid - start)
        y[mid:end] = [class_idx * 2 + 2] * (end - mid)  # plus 1 because no-class is 0
    return y


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
            assert len(test_x.shape) == 2, test_x.shape
            self.data.append({
                'x': test_x,
                'y': test_y,
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
                'dropout': 0.25,
                'example_input_array': [1, 100, 12]  # batch, window, feature_dim
            },
            'optimizer': {
                'lr': 1e-3,
            },
            'lr_scheduler': {
                'step': 10,
                'gamma': 0.1
            },
        })
    return cfg


def train():
    cfg = load_config()
    pl.seed_everything(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = LitModel(cfg)

    LOGGER = [
        CSVLogger(save_dir=cfg.trainer.default_root_dir, name='csv'),
        TensorBoardLogger(save_dir=cfg.trainer.default_root_dir, name='tensorboard'),
        # WandbLogger(save_dir=cfg.trainer.default_root_dir, project='time_series')
    ]
    CALLBACKS = [
        LearningRateMonitor(log_momentum=True),
        ModelCheckpoint(dirpath=cfg.trainer.default_root_dir,
                        filename="acc_{val/acc:.3f}_epoch_{epoch:03d}",
                        monitor='val/acc',
                        mode='min',
                        auto_insert_metric_name=False)
    ]

    trainer = Trainer(
        **cfg.trainer,
        callbacks=CALLBACKS,
        logger=LOGGER,
        # fast_dev_run=True,
        # strategy=DDPStrategy(find_unused_parameters=True, process_group_backend='gloo'),
    )
    # trainer.fit(model)
    trainer.test(model)


if __name__ == '__main__':
    # train()
    cfg = load_config()
    pl.seed_everything(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = LitModel(cfg)
    model.load_from_checkpoint('exp/time_series/acc_0.598_epoch_000-v1.ckpt')

    trainer = Trainer(**cfg.trainer)
    trainer.test(model)