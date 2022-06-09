import argparse
import cv2
from WorkoutDetector.datasets import RepcountDataset, RepcountImageDataset
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import os
from torch import optim, nn, utils, Tensor
import pytorch_lightning as pl
from torchvision import models
import torchvision.transforms as T
from pytorch_lightning.utilities.cli import LightningCLI
import timm
import yaml

proj_config = yaml.safe_load(
    open(os.path.join(os.path.dirname(__file__), 'utils/config.yml')))
proj_root = proj_config['proj_root']

data_transforms = {
    'train':
        T.Compose([
            T.ToPILImage(),
            T.Resize(256),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    'val':
        T.Compose([
            T.ToPILImage(),
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
}


class MyModel(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(),
                                    nn.Linear(128, 1))

    def forward(self, x):
        return self.layers(x)


class LitImageModel(pl.LightningModule):

    def __init__(self, backbone_model, learning_rate):
        super().__init__()
        self.example_input_array = torch.randn(1, 3, 224, 224)
        self.save_hyperparameters()
        # init a pretrained resnet
        backbone = timm.create_model(backbone_model, pretrained=True, num_classes=2)
        self.classifier = backbone
        self.loss_module = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_module(y_hat, y)
        # classes = (y_hat > 0.5).float()
        # acc = (classes == y).float().mean()
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("train/acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train/loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_module(y_hat, y)
        # classes = (y_hat > 0.5).float()
        # acc = (classes == y).float().mean()
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("val/acc", acc, on_step=False, on_epoch=True)
        self.log('val/loss', loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_module(y_hat, y)
        # classes = (y_hat > 0.5).float()
        # acc = (classes == y).float().mean()
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("test/acc", acc, on_step=False, on_epoch=True)
        self.log('test/loss', loss, prog_bar=True)

    def configure_optimizers(self):
        # return optim.AdamW(self.parameters(), lr=self.hparams.lr)
        optimizer = optim.SGD(self.parameters(),
                              lr=self.learning_rate,
                              momentum=0.9,
                              weight_decay=0.0001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return [optimizer], [scheduler]

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)


def main():
    batch_size = 32
    lr = 5e-4
    epochs = 10
    pl.seed_everything(0)

    action = 'situp'
    wandb_logger = pl.loggers.WandbLogger(project="binary-action-classification",
                                          name=action)
    backbone = 'mobilenetv3_small_050'
    hparams = {
        'accelerator': 'gpu',
        'action': action,
        'batch_size': batch_size,
        'lr': lr,
        'max-epochs': epochs
    }
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False
    data_root = os.path.join(proj_root, 'data')
    train_set = RepcountImageDataset(root=data_root,
                                     action=action,
                                     split='train',
                                     transform=data_transforms['train'])
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)
    val_set = RepcountImageDataset(root=data_root,
                                   action=action,
                                   split='val',
                                   transform=data_transforms['val'])
    val_loader = DataLoader(val_set,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True)
    test_set = RepcountImageDataset(root=data_root,
                                    action=action,
                                    split='test',
                                    transform=data_transforms['val'])
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True)
    print(len(train_set), len(val_set), len(test_set))

    model = LitImageModel(backbone, lr)
    wandb_logger.watch(model, log="all")
    trainer = pl.Trainer(hparams)
    trainer.fit(model, train_loader, val_loader)


def export_model(ckpt):
    model = LitImageModel.load_from_checkpoint(ckpt)
    model.eval()
    onnx_path = ckpt.replace('.ckpt', '.onnx')
    model.to_onnx(onnx_path, export_params=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='train or export')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to load')
    args = parser.parse_args()
    if args.train:
        main()
    else:
        export_model(args.ckpt)