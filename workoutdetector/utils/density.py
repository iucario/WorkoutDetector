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

import os
from typing import Any, Callable, List, Tuple, Union

import torch
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
        if split == 'test':
            self.data_list, self.label_list = self._test_dataset()
        else:
            self.data_list, self.label_list = self._train_dataset()
            for x in self.data_list:
                assert x.dtype == torch.float32, x.dtype
            for x in self.label_list:
                assert x.dtype == torch.float32, x.dtype

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
            pkl = torch.load(path)
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
        if self.split == 'test':
            return self._get_test_item(index)
        return self.data_list[index], self.label_list[index]

    def __len__(self):
        return len(self.data_list)

    def _get_test_item(self, index) -> Tuple[Tensor, Tensor, int]:
        """Returns data, normalized label and count."""
        item = self.data_list[index]
        filename = self.filename_tmpl.format(name=item.video_name,
                                             stride=self.stride,
                                             step=self.step)
        path = os.path.join(self.data_root, filename)
        pkl = torch.load(path)
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


def evaluate(model, pkl_path, gt_count: int, device) -> float:
    model.eval()
    model = model.to(device)
    pkl = torch.load(pkl_path).to(device)
    density = 0.0
    for i, x in enumerate(pkl):
        y_pred = model(x.unsqueeze(0))
        density += y_pred.squeeze(1).item()

    return density


def test_epoch(model, loader, loss_fn, device) -> Tuple[float, float]:
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
    files = [f for f in os.listdir(dir_path) if f.endswith('.pkl')]
    data = torch.load(os.path.join(dir_path, files[0]))
    print(len(data), data[0].shape)
    anno_path = os.path.expanduser('~/data/RepCount/annotation.csv')
    val_ds = Dataset(data_root=dir_path, anno_path=anno_path, split='val')
    test_ds = Dataset(data_root=dir_path, anno_path=anno_path, split='test')
    print(len(val_ds), len(test_ds))

    device = 'cuda:7' if torch.cuda.is_available() else 'cpu'
    model = Regressor(input_dim=2048, output_dim=1).to(device)
    loss_fn = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=1e-6)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    train_loader = DataLoader(val_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    for epoch in range(100):
        train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device)
        test_epoch(model, test_loader, loss_fn, device)


if __name__ == '__main__':
    main()
