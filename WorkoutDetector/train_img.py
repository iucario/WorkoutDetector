from utils.datasets import ImageDataset
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
import pytorch_lightning as pl


train_set = ImageDataset(classname='squat', split='train')
train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
print(len(train_set))

