from utils.datasets import ImageDataset
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
import pytorch_lightning as pl
from torchvision import models
import torchvision.transforms as T


data_transforms = {
    'train': T.Compose([
        T.ToPILImage(),
        T.Resize(256),
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': T.Compose([
        T.ToPILImage(),
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class ImagenetTransferLearning(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # init a pretrained resnet
        backbone = models.resnet18(pretrained=True)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        num_target_classes = 2
        self.classifier = nn.Linear(num_filters, num_target_classes)
        self.loss_module = nn.CrossEntropyLoss()


    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_module(y_hat, y)
        acc = (y_hat.argmax(dim=-1) == y).float().mean()
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_module(y_hat, y)
        acc = (y_hat.argmax(dim=-1) == y).float().mean()
        self.log("val_acc", acc, on_step=False, on_epoch=True)
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_module(y_hat, y)
        acc = (y_hat.argmax(dim=-1) == y).float().mean()
        self.log("test_acc", acc, on_step=False, on_epoch=True)
        self.log('test_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)


if __name__ == '__main__':
    batch_size = 32
    pl.seed_everything(42)
    train_set = ImageDataset(classname='squat', split='train',
                             transform=data_transforms['train'])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_set = ImageDataset(classname='squat', split='val',
                           transform=data_transforms['val'])
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    test_set = ImageDataset(classname='squat', split='test',
                            transform=data_transforms['val'])
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    print(len(train_set), len(val_set), len(test_set))

    model = ImagenetTransferLearning()
    trainer = pl.Trainer(limit_train_batches=100, max_epochs=10,
                         default_root_dir='.', accelerator="gpu", devices=1)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader,)
    val_result = trainer.test(val_dataloaders=val_loader)
    test_result = trainer.test(test_dataloaders=test_loader)
    print(val_result)
    print(test_result)