import os

os.environ["NCCL_DEBUG"] = "INFO"
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
os.environ['NCCL_P2P_LEVEL'] = "LOC"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'


import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.datasets import MNIST, CIFAR10
import timm
from torchvision.models import resnet18, vgg16
from torch.utils.data import DataLoader, random_split, Subset


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
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class VGG16(nn.Module):

    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()

        self.features = nn.Sequential(
            self.block(3, 64),
            nn.MaxPool2d(2, 2),
            self.block(64, 128),
            nn.MaxPool2d(2, 2),
            self.block(128, 256),
            nn.MaxPool2d(2, 2),
            self.block(256, 512),
            nn.MaxPool2d(2, 2),
            self.block(512, 512),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def block(self, i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3), nn.ReLU(inplace=True))


class MyModel(nn.Module):

    def __init__(self, dim: int = 2):
        super(MyModel, self).__init__()
        # fx = resnet18(pretrained=True)
        fx = vgg16()
        # fx = timm.create_model('resnet18', pretrained=False)
        in_features = fx.classifier[0].in_features
        fx.fc = nn.Linear(in_features, dim)
        self.model = fx

    def forward(self, x):
        return self.model(x)


class LitModel(pl.LightningModule):

    def __init__(self, dim_out: int = 10):
        super().__init__()
        # fx = resnet18(pretrained=True)
        fx = vgg16()
        # fx = timm.create_model('resnet18', pretrained=False)
        in_features = fx.classifier[0].in_features
        fx.fc = nn.Linear(in_features, dim_out)
        self.model = fx
        
        # self.model = CNN(dim_out)
        # self.model = VGG16(dim_out)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.model(x)
        loss = F.cross_entropy(z, y)
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class DataModule(pl.LightningDataModule):

    def __init__(self, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size
        transforms = T.Compose([
            T.ToTensor(),
            T.Resize(size=(224, 224)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.dataset = CIFAR10(root='./data',
                               train=True,
                               transform=transforms,
                               download=True)

    def train_dataloader(self):
        return DataLoader(Subset(self.dataset, list(range(200))),
                          batch_size=self.batch_size,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(Subset(self.dataset, list(range(200, 300))),
                          batch_size=self.batch_size,
                          shuffle=True)


def run():

    dataset = DataModule(2)
    train_loader = dataset.train_dataloader()
    print(len(train_loader))
    print(next(iter(train_loader))[0].shape, next(iter(train_loader))[1])
    model = LitModel(dim_out=10)

    x = torch.randn(32, 3, 224, 224)
    y = model(x)
    print(y.shape)

    trainer = pl.Trainer(
        default_root_dir=os.getcwd(),
        fast_dev_run=False,
        accelerator="gpu",
        devices="auto",
        strategy="ddp",
        # find_unused_parameters=False,
        max_epochs=3,
    )
    trainer.fit(model=model, train_dataloaders=train_loader)


if __name__ == "__main__":
    run()