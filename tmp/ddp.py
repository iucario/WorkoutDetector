import os
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torchvision.datasets import MNIST, CIFAR10
import timm
from torchvision.models import resnet18, vgg16
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

os.environ["NCCL_DEBUG"] = "INFO"
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
os.environ['NCCL_P2P_LEVEL'] = 'LOC'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class CNN(nn.Module):

    def __init__(self, num_classes=10):
        super(CNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3),
            nn.MaxPool2d(2, 2),
            # self.block(256, 512),
            # nn.MaxPool2d(2, 2),
            # self.block(512, 512),
            # nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 100),
            # nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ToyModel(nn.Module):

    def __init__(self, dim: int = 2):
        super(ToyModel, self).__init__()
        self.model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 224 * 224, 1000),
                                   *[nn.Linear(1000, 1000)] * 20, nn.Linear(1000, dim))

    def forward(self, x):
        return self.model(x)


class Net(nn.Module):

    def __init__(self, num_class: int = 10):
        super(Net, self).__init__()
        fx = resnet18(pretrained=True)
        fx.fc = nn.Linear(512, num_class)
        self.model = fx

    def forward(self, x):
        return self.model(x)


def example(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # model = resnet18(weights='ResNet18_Weights.IMAGENET1K_V1').to(rank)
    model = Net(10).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    outputs = ddp_model(torch.randn(8, 3, 224, 224).to(rank))
    labels = torch.randn(8, 10).to(rank)
    loss_fn(outputs, labels).backward()
    optimizer.step()


def main():
    world_size = 2
    print("We have available ", torch.cuda.device_count(), "GPUs! Using ", world_size,
          " GPUs")

    y = Net(10)(torch.randn(1, 3, 224, 224))
    print(y.shape)

    mp.spawn(example, args=(world_size,), nprocs=world_size, join=True)
    print('Success!')


if __name__ == "__main__":
    # Environment variables which need to be
    # set when using c10d's default "env"
    # initialization mode.
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    main()
