import os

os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['NCCL_DEBUG_SUBSYS'] = 'ENV'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
os.environ['NCCL_P2P_LEVEL'] = 'LOC'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torch.utils.data import random_split, DataLoader, Subset
from torchvision.datasets import CIFAR10, MNIST
from torchvision.models import resnet18

torch.autograd.set_detect_anomaly(True)


class Net(nn.Module):

    def __init__(self, num_class: int = 10):
        super(Net, self).__init__()
        fx = resnet18(pretrained=True)
        fx.fc = nn.Linear(512, num_class)
        self.model = fx

    def forward(self, x):
        return self.model(x)


def train(model, device, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        print(f'{batch_idx}/{len(train_loader)}, loss={loss.item():.4f}')


def main():
    device = 'cuda'

    dataset = CIFAR10('./data',
                      'train',
                      download=True,
                      transform=T.Compose([
                          T.ToTensor(),
                          T.Resize(size=(224, 224)),
                          T.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]),
                      ]))
    train_set = Subset(dataset, range(200))
    train_loader = DataLoader(train_set, batch_size=20, shuffle=True, num_workers=1)
    print(dataset[0][0].shape)

    model = Net(num_class=10)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model).to(device)

    y = model(torch.randn(1, 3, 224, 224).to(device))
    print(y.shape)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    print('start training')
    for i in range(3):
        train(model, device, train_loader, optimizer)

    print('Done')


if __name__ == '__main__':
    main()