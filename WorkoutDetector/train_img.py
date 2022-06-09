import cv2
from utils.datasets import ImageDataset, SuperImageDataset
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
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_module(y_hat, y)
        # classes = (y_hat > 0.5).float()
        # acc = (classes == y).float().mean()
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("val_acc", acc, on_step=False, on_epoch=True)
        self.log('val_loss', loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_module(y_hat, y)
        # classes = (y_hat > 0.5).float()
        # acc = (classes == y).float().mean()
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("test_acc", acc, on_step=False, on_epoch=True)
        self.log('test_loss', loss, prog_bar=True)


    def configure_optimizers(self):
        # return optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optim.SGD(self.parameters(),
                         lr=self.learning_rate,
                         momentum=0.9,
                         weight_decay=0.0001)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)


def train_model(model, train_loader, val_loader, test_loader, epochs=20):
    trainer = pl.Trainer(limit_train_batches=100,
                         max_epochs=epochs,
                         default_root_dir='.',
                         accelerator="gpu",
                         devices=1)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    val_result = trainer.test(model, dataloaders=val_loader)
    test_result = trainer.test(model, dataloaders=test_loader)
    print(val_result)
    print(test_result)


def export_model(ckpt):
    model = LitImageModel.load_from_checkpoint(ckpt)
    model.eval()
    onnx_path = 'lightning_logs/version_2/squat.onnx'
    model.to_onnx(onnx_path, export_params=True)

def infer_video(model, video):
    model.eval()
    cap = cv2.VideoCapture(video)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (224, 224))
            frame = T.ToTensor()(frame)
            frame = frame.unsqueeze(0)
            frame = frame.to(torch.device('cuda'))
            output = model(frame)
            output = output.detach().cpu().numpy()
            print(output[0])
        else:
            break
    cap.release()

if __name__ == '__main__':
    batch_size = 32
    lr = 5e-5
    epochs = 10
    pl.seed_everything(0)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False
    train_set = ImageDataset(classname='squat',
                             split='train',
                             transform=data_transforms['train'])
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True)
    val_set = ImageDataset(classname='squat',
                           split='val',
                           transform=data_transforms['val'])
    val_loader = DataLoader(val_set,
                            batch_size=batch_size,
                            shuffle=False,
                            pin_memory=True)
    test_set = ImageDataset(classname='squat',
                            split='test',
                            transform=data_transforms['val'])
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             pin_memory=True)
    print(len(train_set), len(val_set), len(test_set))

    model = LitImageModel('mobilenetv2_050', lr)
    # trainer = pl.Trainer(accelerator='gpu')
    # trainer.fit(model, train_loader, val_loader)

    # train_model(model, train_loader, val_loader, test_loader, epochs=epochs)

    # infer_video(model, 'squat.mp4')
