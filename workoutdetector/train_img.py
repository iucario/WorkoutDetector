import argparse
import os
import os.path as osp
from os.path import join as osj
from typing import Tuple, List, Optional, Callable

import pytorch_lightning as pl
import timm
import torch
import torchvision.transforms as T
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, early_stopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch import Tensor, nn, optim, utils
from torchvision.io import read_image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from workoutdetector.datasets import RepcountImageDataset
from workoutdetector.settings import PROJ_ROOT

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
    'test':
        T.Compose([
            T.ToPILImage(),
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
}


def get_data_loaders(action: str,
                     batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    data_root = os.path.join(PROJ_ROOT, 'data')
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
    return train_loader, val_loader, test_loader


class Detector():

    def __init__(self):
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        self.model.cuda()

    def detect(self, image: Tensor, threshold: float = 0.7) -> Tensor:
        result: List[dict] = self.model(image)
        person_boxes = result[0]['boxes'][result[0]['labels'] == 1]
        scores = result[0]['scores'][result[0]['labels'] == 1]
        scores = scores[scores > threshold]
        person_boxes = person_boxes[scores > threshold]
        x1, y1, x2, y2 = person_boxes[0].cpu().numpy()
        return person_boxes


class MyModel(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(),
                                    nn.Linear(128, 1))

    def forward(self, x):
        return self.layers(x)


class LitImageModel(LightningModule):

    def __init__(self, num_class: int, backbone_model, learning_rate):
        super().__init__()
        self.example_input_array = torch.randn(1, 3, 224, 224)
        self.save_hyperparameters()
        # init a pretrained resnet
        backbone = timm.create_model(backbone_model,
                                     pretrained=True,
                                     num_classes=num_class)
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
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_module(y_hat, y)
        # classes = (y_hat > 0.5).float()
        # acc = (classes == y).float().mean()
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("val/acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val/loss', loss)

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
        return optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        optimizer = optim.SGD(self.parameters(),
                              lr=self.learning_rate,
                              momentum=0.9,
                              weight_decay=0.0001)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val/loss",
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)


class DataModule(LightningDataModule):

    def __init__(self, data_root: str, batch_size: int, num_workers: int = 4) -> None:
        super().__init__()
        self.prepare_data_per_node = False
        self.batch_size = num_workers
        self.num_workers = 4
        self.train_set, self.val_set, self.test_set = [
            ImageFolder(
                os.path.join(data_root, split),
                transform=data_transforms['train' if split == 'train' else 'test'])
            for split in ('train', 'val', 'test')
        ]

    def train_dataloader(self):
        loader = DataLoader(self.train_set,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            shuffle=True)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.val_set,
                            num_workers=self.num_workers,
                            batch_size=self.batch_size,
                            shuffle=False)
        return loader

    def test_dataloader(self):
        loader = DataLoader(self.test_set,
                            num_workers=self.num_workers,
                            batch_size=self.batch_size,
                            shuffle=False)
        return loader


class NoSplitData(LightningDataModule):

    def __init__(self, data_root: str, batch_size: int, num_workers: int = 4) -> None:
        super().__init__()
        self.prepare_data_per_node = False
        self.batch_size = num_workers
        self.num_workers = 4
        dataset = ImageFolder(data_root)
        ratio = [int(len(dataset) * 0.7), len(dataset) - int(len(dataset) * 0.7)]
        self.train_set, self.val_set = torch.utils.data.random_split(
            dataset, ratio, generator=torch.Generator().manual_seed(42))

    def train_dataloader(self):
        loader = DataLoader(self.train_set,
                            num_workers=self.num_workers,
                            batch_size=self.batch_size,
                            shuffle=True)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.val_set,
                            num_workers=self.num_workers,
                            batch_size=self.batch_size,
                            shuffle=False)
        return loader

    def test_dataloader(self):
        loader = DataLoader(self.val_set,
                            num_workers=self.num_workers,
                            batch_size=self.batch_size,
                            shuffle=False)
        return loader


class ImageDataset(torch.utils.data.Dataset):
    """General image dataset
    label text files of [image.png class] are required

    Args:
        data_root: str
        data_prefix: str, will be appended to data_root
        anno_path: str, abusolute annotation path
        transform: Optional[Callable], data transform
    """

    def __init__(self,
                 data_root: str,
                 data_prefix: str,
                 anno_path: str = 'train.txt',
                 transform: Optional[Callable] = None) -> None:
        super().__init__()
        assert osp.isfile(anno_path), f'{anno_path} is not file'
        self.data_root = data_root
        self.data_prefix = osj(data_root, data_prefix)
        self.transform = transform
        self.anno: List[Tuple[str, int]] = self.read_txt(anno_path)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        path, label = self.anno[index]
        img = read_image(osj(self.data_prefix, path))
        if self.transform:
            img = self.transform(img)
        return img, int(label)

    def __len__(self) -> int:
        return len(self.anno)

    def read_txt(self, path: str) -> List[Tuple[str, int]]:
        """Read annotation file 
        
        Args:
            path: str, path to annotation file

        Returns:
            List of [path, class]
        """
        ret = []
        with open(path) as f:
            for line in f:
                name, class_ = line.strip().split()
                ret.append((name, int(class_)))
        return ret


class ImageDataModule(LightningDataModule):
    """General image dataset
    label text files of [image.png class] are required
    """

    def __init__(self,
                 data_root: str,
                 batch_size: int,
                 train_anno: str = 'train.txt',
                 val_anno: str = 'val.txt',
                 test_anno: Optional[str] = None,
                 train_data_prefix: Optional[str] = None,
                 val_data_prefix: Optional[str] = None,
                 test_data_prefix: Optional[str] = None,
                 num_workers: int = 4,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        super().__init__()
        assert osp.exists(osj(data_root,
                              train_anno)), f'{osj(data_root, train_anno)} does not exist'
        assert osp.exists(osj(data_root,
                              val_anno)), f'{osj(data_root, val_anno)} does not exist'
        if test_anno:
            assert osp.exists(osj(
                data_root, test_anno)), f'{osj(data_root, test_anno)} does not exist'
        self.prepare_data_per_node = False
        self.data_root = data_root
        self.batch_size = num_workers
        self.num_workers = num_workers
        self.train_anno = train_anno
        self.val_anno = val_anno
        self.test_anno = test_anno
        self.train_data_prefix = train_data_prefix
        self.val_data_prefix = val_data_prefix
        self.test_data_prefix = test_data_prefix
        self.transform = transform
        self.target_transform = target_transform

    def train_dataloader(self):
        train_set = ImageDataset(self.data_root, self.train_data_prefix, self.train_anno,
                                 self.transform)
        loader = DataLoader(train_set,
                            num_workers=self.num_workers,
                            batch_size=self.batch_size,
                            shuffle=True)
        return loader

    def val_dataloader(self):
        val_set = ImageDataset(self.data_root, self.val_data_prefix, self.val_anno,
                               self.transform)
        loader = DataLoader(val_set,
                            num_workers=self.num_workers,
                            batch_size=self.batch_size,
                            shuffle=False)
        return loader

    def test_dataloader(self):
        if self.test_anno:
            test_set = ImageDataset(self.data_root, self.test_data_prefix, self.test_anno,
                                    self.transform)
            loader = DataLoader(test_set,
                                num_workers=self.num_workers,
                                batch_size=self.batch_size,
                                shuffle=False)
            return loader
        else:
            return self.val_dataloader()


def export_model(ckpt):
    model = LitImageModel.load_from_checkpoint(ckpt)
    model.eval()
    onnx_path = ckpt.replace('.ckpt', '.onnx')
    model.to_onnx(onnx_path, export_params=True)


def main(args):
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    backbone = args.backbone

    pl.seed_everything(0)
    data_root = os.path.expanduser('~/data/situp')
    # data_module = DataModule(data_root, batch_size)
    data_module = ImageDataModule(data_root,
                                  batch_size,
                                  train_anno=osj(data_root, 'train.txt'),
                                  val_anno=osj(data_root, 'val.txt'),
                                  test_anno=None,
                                  train_data_prefix='train',
                                  val_data_prefix='val',
                                  test_data_prefix=None,
                                  num_workers=4,
                                  transform=data_transforms['train'],
                                  target_transform=data_transforms['test'])
    # print(data_module.train_dataloader())
    # exit(1)
    # callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stop = early_stopping.EarlyStopping(monitor='train/loss',
                                              mode='min',
                                              patience=10)
    # loggers
    # wandb_logger = WandbLogger(
    #     save_dir=os.path.join(PROJ_ROOT, 'log/image_model'),
    #     project="binary-action-classification",
    # )
    # wandb_logger.log_hyperparams(args)
    tensorboard_logger = TensorBoardLogger(save_dir=os.path.join(
        PROJ_ROOT, f'log/{args.project}/tensorboard'),
                                           default_hp_metric=False)
    tensorboard_logger.log_hyperparams(args,
                                       metrics={
                                           'train/acc': 0,
                                           'val/acc': 0,
                                           'test/acc': 0,
                                           'train/loss': 1,
                                           'val/loss': 1,
                                           'test/loss': 1
                                       })

    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    model = LitImageModel(num_class=args.num_class,
                          backbone_model=backbone,
                          learning_rate=lr)
    # wandb_logger.watch(model, log="all")

    trainer = Trainer(
        default_root_dir=os.path.join(PROJ_ROOT, f'log/{args.project}'),
        max_epochs=epochs,
        accelerator='gpu',
        devices=1,
        logger=[tensorboard_logger],
        callbacks=[lr_monitor, early_stop],
        auto_lr_find=True,
    )
    # lr_finder = trainer.tuner.lr_find(model, train_loader, val_loader)
    # new_lr = lr_finder.suggestion()
    # model.hparams.learning_rate = new_lr

    trainer.fit(model, data_module)
    trainer.test(model, data_module)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ck',
                        '--ckpt',
                        type=str,
                        default=None,
                        help='checkpoint to load')
    parser.add_argument('-lr', '--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('-e', '--epochs', type=int, default=20, help='epochs')
    parser.add_argument('-b',
                        '--backbone',
                        type=str,
                        default='convnext_base',
                        help='backbone')
    parser.add_argument('-bs', '--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--num-class', type=int, default=2, help='number of classes')
    parser.add_argument('--project',
                        type=str,
                        default='situp-image-selected',
                        help='project name. Effects logger dirs')
    args = parser.parse_args()
    main(args)
