from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import random_split, DataLoader, Subset
from torchvision import transforms as T
from torchvision.datasets import MNIST


class DataParallelModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.block1 = nn.Linear(28 * 28, 100)
        self.block2 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.block1(x)
        x = self.block2(x)
        return x


class Model(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.fc = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        return self.fc(x)


def train_epoch(model, batch, loss_fn, optimizer, device):
    model.train()
    optimizer.zero_grad()
    x, y = batch
    x = x.view(x.size(0), -1).to(device)
    y = y.to(device)
    z = model(x)
    loss = loss_fn(z, y)
    loss.backward()
    optimizer.step()
    return loss.item(), (z.argmax(dim=1) == y).sum().item()


def val_epoch(model, batch, loss_fn, device):
    model.eval()
    with torch.no_grad():
        x, y = batch
        x = x.view(x.size(0), -1).to(device)
        y = y.to(device)
        z = model(x)
        loss = loss_fn(z, y)
    return loss.item(), (z.argmax(dim=1) == y).sum().item()


def main():
    device = 'cpu'

    dataset = MNIST('./data',
                    'train',
                    download=True,
                    transform=T.Compose([T.ToTensor(),
                                         T.Normalize((0.1307,), (0.3081,))]))
    train_set, val_set = random_split(Subset(dataset, range(500)), [400, 100])
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)
    print(dataset[0][0].shape)

    model = Model(28*28, 10)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for i in range(10):
        train_correct = 0
        for batch in train_loader:
            loss, correct = train_epoch(model, batch, loss_fn, optimizer, device)
            train_correct += correct
        train_acc = train_correct / len(train_loader.dataset)
        print(f"Train loss: {loss:.4f}, Train acc: {train_acc:.4f}")
        val_correct = 0
        for batch in val_loader:
            loss, correct = val_epoch(model, batch, loss_fn, device)
            val_correct += correct
        val_acc = val_correct / len(val_loader.dataset)
        print(f"Val loss: {loss:.4f}, Val acc: {val_acc:.4f}")


if __name__ == '__main__':
    main()