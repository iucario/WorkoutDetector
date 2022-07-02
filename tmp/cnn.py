import os

os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['NCCL_DEBUG_SUBSYS'] = 'ENV'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
os.environ['NCCL_P2P_LEVEL'] = 'LOC'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import random_split, DataLoader, Subset
from torchvision import transforms as T
from torchvision.datasets import CIFAR10, MNIST
from torchvision.models import resnet50


def train_epoch(model, batch, loss_fn, optimizer, device):
    model.train()
    optimizer.zero_grad()
    x, y = batch
    x = x.to(device)
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
        x = x.to(device)
        y = y.to(device)
        z = model(x)
        loss = loss_fn(z, y)
    return loss.item(), (z.argmax(dim=1) == y).sum().item()


def main():
    device = 'cuda:0'

    dataset = CIFAR10('./data',
                      'train',
                      download=True,
                      transform=T.Compose([
                          T.ToTensor(),
                          T.Resize(size=(224, 224)),
                      ]))
    train_set, val_set = random_split(Subset(dataset, range(500)), [400, 100])
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)
    print(dataset[0][0].shape)

    model = resnet50(pretrained=True)
    model = model.to(device)

    y = model(torch.randn(1, 3, 224, 224).to(device))
    print(y.shape)

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
