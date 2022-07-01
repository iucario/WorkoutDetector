import os

os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['NCCL_DEBUG_SUBSYS'] = 'ENV'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
os.environ['NCCL_P2P_LEVEL'] = 'LOC'
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

torch.autograd.set_detect_anomaly(True)

class DummyDataset(torch.utils.data.Dataset):

    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.randn(10), 0


class BoringModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.block = nn.Linear(10, 10)

    def forward(self, x):
        x = self.block(x)
        return x

def train(model, device, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward() # This will cause the program to hang
        optimizer.step()
        print(f'{batch_idx}/{len(train_loader)}, loss={loss.item():.4f}')


def main():
    device = 'cuda'

    train_set = DummyDataset(100)
    train_loader = DataLoader(train_set, batch_size=10, pin_memory=True, num_workers=1)

    model = BoringModel()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    print('start training')
    for i in range(1):
        train(model, device, train_loader, optimizer)

    print('Done')


if __name__ == '__main__':
    main()