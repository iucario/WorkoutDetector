"""
0, 0, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 

Guassian density function applied to repetition instances.
    \mu = (start_time + end_time) / 2
    \sigma = end_time - start_time

Inputs:
    Continuous 8 frames, [i, i+1, i+2, i+3, i+4, i+5, i+6, i+7]

Outputs:
    A float value between 0 and 1.

Model:
    A action recognizer + FC layer.

Loss function:
    Mean squared error. Compares with the density function at i+4.
    Formally, the loss is: `nn.MSE(model(i), density(i+4))`

Inference:
    Sum the predicted density at each frame.

Why this method:
    Reduces a loooot of engineering. And is suitable for online inference.
"""

import json
import os
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
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from workoutdetector.datasets.repcount_dataset import RepcountHelper


def density_fn(start: int, end: int) -> Any:
    mid = (end + start) / 2
    sigma = (end - start) / 6
    dist = norm(loc=mid, scale=sigma)
    return dist


def create_label(reps: List[int], total_frames: int) -> Tensor:
    """Create normalized label.

    Args:
        reps (List[int]): A list of repetition starts and ends. `[start_1, end_1, start_2, end_2, ...]`
        total_frames (int): Total number of frames.
    Returns:
        Tensor, shape [total_frames]
    """
    labels = [0] * total_frames
    for s, e in zip(reps[::2], reps[1::2]):
        dist = density_fn(s, e)
        for i in range(s, min(e + 1, total_frames)):
            labels[i] = dist.pdf(i)
    return torch.tensor(labels, dtype=torch.float32)


class Dataset(torch.utils.data.Dataset):

    def __init__(self,
                 data_root: str,
                 anno_path: str,
                 split: str,
                 num_frames: int = 8,
                 stride: int = 1,
                 step: int = 2,
                 filename_tmpl: str = '{name}.stride_{stride}_step_{step}.pkl'):
        self.data_root = data_root
        self.anno_path = anno_path
        self.split = split
        self.stride = stride
        self.step = step
        self.filename_tmpl = filename_tmpl
        helper = RepcountHelper(os.path.expanduser('~/data/RepCount'), anno_path)
        self.data = helper.get_rep_data(split=[split], action=['all'])
        if split == 'train':
            self.data_list, self.label_list = self._train_dataset()
            for x in self.data_list:
                assert x.dtype == torch.float32, x.dtype
            for x in self.label_list:
                assert x.dtype == torch.float32, x.dtype
        else:
            self.data_list, self.label_list = self._test_dataset()

    def _train_dataset(self) -> Tuple[list, list]:
        # Use all items in pkl
        data_list: List[Tensor] = []
        label_list: List[float] = []
        for item in self.data.values():
            filename = self.filename_tmpl.format(name=item.video_name,
                                                 stride=self.stride,
                                                 step=self.step)
            path = os.path.join(self.data_root, filename)
            # list of Tensor(8, 2048). Features of [i:i+8)
            pkl = torch.load(path, map_location='cpu')
            norm_label = create_label(item.reps, item.total_frames)
            data_list += pkl
            # Align data[i:i+8*step] with label[i+4*step]
            label_list += norm_label[4 * self.step:4 * self.step + len(pkl):self.stride]
        assert len(data_list) == len(label_list)
        return data_list, label_list

    def _test_dataset(self) -> Tuple[list, list]:
        data_list = list(self.data.values())
        label_list = [item.count for item in data_list]
        return data_list, label_list

    def __getitem__(self,
                    index) -> Union[Tuple[Tensor, float], Tuple[Tensor, Tensor, int]]:
        if self.split == 'train':
            return self.data_list[index], self.label_list[index]
        return self._get_test_item(index)

    def __len__(self):
        return len(self.data_list)

    def _get_test_item(self, index) -> Tuple[Tensor, Tensor, int]:
        """Returns data, normalized label and count."""
        item = self.data_list[index]
        filename = self.filename_tmpl.format(name=item.video_name,
                                             stride=self.stride,
                                             step=self.step)
        path = os.path.join(self.data_root, filename)
        pkl = torch.load(path, map_location='cpu')
        norm_label = create_label(item.reps, item.total_frames)
        norm_label = norm_label[4 * self.step:4 * self.step + len(pkl):self.stride]
        return torch.stack(pkl), norm_label, item.count


class Regressor(nn.Module):
    """Regressor for predicting density."""

    def __init__(self,
                 input_dim: int = 2048,
                 output_dim: int = 1,
                 hidden_dim: int = 512,
                 dropout: float = 0.25):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        # [batch, num_frames, input_dim]
        x = x.mean(dim=1)  # [batch, input_dim]
        return self.layers(x)


class LitModel(LightningModule):

    def __init__(self, cfg: CfgNode):
        super().__init__()
        self.save_hyperparameters(cfg.__dict__)
        if self.logger:
            self.logger.log_hyperparams(cfg.__dict__)
        self.model = Regressor(input_dim=cfg.model.input_dim,
                               output_dim=cfg.model.output_dim,
                               hidden_dim=cfg.model.hidden_dim,
                               dropout=cfg.model.dropout)
        self.loss_fn = nn.MSELoss()
        self.cfg = cfg
        self.data_root = cfg.data.data_root
        self.anno_path = cfg.data.anno_path
        self.example_input_array = torch.randn(1, 8, 2048)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.cfg.optimizer.lr)
        scheduler = lr_scheduler.StepLR(optimizer,
                                        step_size=self.cfg.lr_scheduler.step,
                                        gamma=self.cfg.lr_scheduler.gamma)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        train_ds = Dataset(self.data_root, self.anno_path, split='train')
        return DataLoader(train_ds, batch_size=self.cfg.data.batch_size, shuffle=True)

    def val_dataloader(self):
        val_ds = Dataset(self.data_root, self.anno_path, split='val')
        return DataLoader(val_ds, batch_size=1, shuffle=False)

    def test_dataloader(self):
        test_ds = Dataset(self.data_root, self.anno_path, split='test')
        return DataLoader(test_ds, batch_size=1, shuffle=False)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)  # [batch, 1]
        loss = self.loss_fn(pred.squeeze(-1), y)
        self.log('train/loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Returns MSE loss and MAE."""
        obo, loss, mae = 0, 0, 0
        for data, y, gt_count in batch:  # loader batch_size = 1
            density = 0.0  # sum of density = predicted count
            for i, x in enumerate(data):
                y_pred = self.model(x)
                loss = self.loss_fn(y_pred.squeeze(-1), y.squeeze(0))
                density += y_pred.squeeze(1).sum().item()
                loss += loss.item()
            diff = abs(gt_count.item() - density)
            mae += diff
            obo += 1 if diff <= 1 else 0
        loss /= len(batch)
        mae /= len(batch)
        self.log('val/error', mae, prog_bar=True, sync_dist=True)
        return loss, mae, obo

    def validation_epoch_end(self, outputs):
        """Returns mean loss and MAE."""
        loss = torch.stack([x[0] for x in outputs]).mean()
        mae = torch.stack([x[1] for x in outputs]).mean()
        self.log_dict({'val/loss': loss, 'val/mae': mae})
        return {'val/loss': loss, 'val/mae': mae}

    def predict_step(self, batch, batch_idx):
        """Returns MSE loss and MAE."""
        count_list, mae_list, obo_list = [], [], []

        for data, y, gt_count in batch:
            density = 0.0
            mae = 0.0
            obo = 0
            for i, x in enumerate(data):
                y_pred = self.model(x)
                density += y_pred.squeeze(1).sum().item()
            diff = abs(gt_count.item() - density)
            mae += diff
            obo += 1 if diff <= 1 else 0
            count_list.append(density)
            mae_list.append(mae)
            obo_list.append(obo)
        return count_list, mae_list, obo_list


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                scheduler, loss_fn: Callable, device: str) -> float:
    model.train()
    epoch_loss = 0.0
    pbar = tqdm(loader, desc='Training')
    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)
        assert x.is_floating_point(), x.type()
        assert y.is_floating_point(), y.type()
        y_pred = model(x)  # [batch, 1]
        loss = loss_fn(y_pred.squeeze(-1), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        epoch_loss += loss.item()
        if pbar.n % 10 == 0:
            pbar.set_postfix(train_step_loss=f'{loss.item():.4f}')

    epoch_loss /= len(loader)
    print(f"train_epoch_loss={epoch_loss:.4f}")
    return epoch_loss


def val_epoch(model, loader, loss_fn, device) -> float:
    model.eval()
    epoch_loss = 0.0
    for x, y in tqdm(loader):
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        loss = loss_fn(y_pred.squeeze(-1), y)
        epoch_loss += loss.item()
    epoch_loss /= len(loader)
    print(f"val_loss={epoch_loss:.4f}")
    return epoch_loss


def test_epoch(model, loader, loss_fn, device) -> Tuple[float, float]:
    """Returns MSE loss and MAE."""
    model.eval()
    epoch_loss = 0.0
    mae = 0.0  # mean absolute error
    for data, y, gt_count in tqdm(loader):  # loader batch_size = 1
        data = data.to(device)
        density = 0.0  # sum of density = predicted count
        for i, x in enumerate(data):
            y_pred = model(x).to(device)
            # print(y_pred.shape, y.shape)
            loss = loss_fn(y_pred.squeeze(-1), y.squeeze(0).to(device))
            density += y_pred.squeeze(1).sum().item()
            epoch_loss += loss.item()
        mae += abs(gt_count.item() - density)
    epoch_loss /= len(loader)
    mae /= len(loader)
    print(f"test_loss={epoch_loss:.4f}, test_mae={mae:.4f}")
    return epoch_loss, mae


def main():
    dir_path = 'out/acc_0.923_epoch_10_20220720-151025_1x2'
    ckpt_dir = 'checkpoints'
    files = [f for f in os.listdir(dir_path) if f.endswith('.pkl')]
    data = torch.load(os.path.join(dir_path, files[0]))
    print(len(data), data[0].shape)
    anno_path = os.path.expanduser('~/data/RepCount/annotation.csv')
    train_ds = Dataset(data_root=dir_path, anno_path=anno_path, split='train')
    val_ds = Dataset(data_root=dir_path, anno_path=anno_path, split='val')
    test_ds = Dataset(data_root=dir_path, anno_path=anno_path, split='test')
    print(len(val_ds), len(test_ds))

    device = 'cuda:7' if torch.cuda.is_available() else 'cpu'
    model = Regressor(input_dim=2048, output_dim=1).to(device)
    loss_fn = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=1e-6)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    best_mae = float('inf')
    for epoch in range(100):
        train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device)
        loss, mae = test_epoch(model, val_loader, loss_fn, device)
        if mae < best_mae:
            best_mae = mae
            torch.save(model.state_dict(),
                       os.path.join(ckpt_dir, f'predictor_epoch_{epoch}_mae_{mae}.pth'))


def load_config():
    cfg = CfgNode(
        init_dict={
            'trainer': {
                'max_epochs': 200,
                'devices': 8,
            },
            'data': {
                'data_root': 'out/acc_0.923_epoch_10_20220720-151025_1x2',
                'anno_path': os.path.expanduser('~/data/RepCount/annotation.csv'),
                'batch_size': 32 * 8,
            },
            'model': {
                'input_dim': 2048,
                'output_dim': 1,
                'hidden_dim': 512,
                'dropout': 0.25,
            },
            'optimizer': {
                'lr': 1e-5 * 8
            },
            'lr_scheduler': {
                'step': 20,
                'gamma': 0.1
            },
        })
    return cfg


def train():
    cfg = load_config()
    pl.seed_everything(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    exp = 'density'
    LOGGER = [
        CSVLogger('exp', name=exp),
        WandbLogger(
            save_dir=f'exp/{exp}',
            project='density',
            offline=True,
        )
    ]
    CALLBACKS = [
        LearningRateMonitor(logging_interval='epoch', log_momentum=True),
        ModelCheckpoint(dirpath=f'exp/{exp}',
                        filename="mae_{val/mae:.3f}_epoch_{epoch:03d}",
                        auto_insert_metric_name=False)
    ]
    model = LitModel(cfg)
    trainer = Trainer(
        model,
        **cfg.trainer,
        fast_dev_run=True,
        logger=LOGGER,
        callbacks=CALLBACKS,
        strategy=DDPStrategy(find_unused_parameters=True, process_group_backend='gloo'),
    )
    trainer.fit(model)


def eval_one_video(model, pkl_path, device) -> float:
    pkl = torch.load(pkl_path, map_location=device)
    density = 0.0
    for i, x in enumerate(pkl):
        y_pred = model(x.unsqueeze(0))
        density += y_pred.squeeze(1).item()
    return density


def evaluate(stride: int = 1):
    split = 'test'
    result = dict()
    ckpt_path = 'checkpoints/predictor_epoch_99_mae_3.7959224247932433.pth'
    device = 'cuda:0'
    ckpt = torch.load(ckpt_path, map_location=device)
    model = Regressor(input_dim=2048, output_dim=1).to(device)
    model.load_state_dict(ckpt)
    model.eval()
    ds = Dataset(data_root='out/acc_0.923_epoch_10_20220720-151025_1x2',
                     anno_path=os.path.expanduser('~/data/RepCount/annotation.csv'),
                     split=split)
    mae, obo = 0.0, 0.0
    for i, (x, y, gt_count) in enumerate(tqdm(ds)):
        x = x.to(device)
        density = 0.0
        for z in x[::stride]:
            y_pred = model(z.unsqueeze(0))
            density += y_pred.squeeze(1).item() * stride
        result[i] = dict(pred_count=density, gt_count=gt_count)
        diff = abs(gt_count - density)
        mae += diff
        obo += 1 if diff <= 1 else 0
    mae /= len(ds)
    obo /= len(ds)
    print(f"stride={stride}, mae={mae:.4f}, obo={obo:.4f}")
    json.dump(result, open(f'out/predictor_{split}_stride_{stride}.json', 'w'))
    return mae


if __name__ == '__main__':
    # train()
    evaluate(stride=8)
