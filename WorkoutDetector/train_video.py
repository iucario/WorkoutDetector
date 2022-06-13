import argparse
from typing import Tuple
from WorkoutDetector.datasets import RepcountVideoDataset
import torch
from torch.utils.data import DataLoader
import os
from torch import optim, nn, utils, Tensor
import pytorch_lightning as pl
import torchvision.transforms as T
import timm
import yaml
import einops
import time

from mmaction.models.backbones import ResNetTSM
from mmaction.models.heads import TSMHead
from mmaction.models import build_model

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


def get_data_loaders(action: str,
                     batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    data_root = os.path.join(proj_root, 'data')
    train_set = RepcountVideoDataset(root=data_root,
                                     action=action,
                                     split='train',
                                     transform=data_transforms['train'])
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)
    val_set = RepcountVideoDataset(root=data_root,
                                   action=action,
                                   split='val',
                                   transform=data_transforms['val'])
    val_loader = DataLoader(val_set,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True)
    test_set = RepcountVideoDataset(root=data_root,
                                    action=action,
                                    split='test',
                                    transform=data_transforms['val'])
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True)
    return train_loader, val_loader, test_loader


class VideoModel(nn.Module):
    """mmaction2 TSM model"""

    def __init__(
        self,
        backbone_model: str = 'resnet18',
        num_classes=2,
        num_segments=8,
    ):
        super().__init__()
        resnet_map = {
            'resnet18': [18, 512],
            'resnet34': [34, 512],
            'resnet50': [50, 2048],
        }
        depth, num_features = resnet_map[backbone_model]
        self.backbone = ResNetTSM(
            depth=depth,
            norm_eval=False,
            shift_div=8,
            pretrained=f'torchvision://{backbone_model}',
        )
        self.head = TSMHead(
            num_classes=num_classes,
            in_channels=num_features,
            num_segments=num_segments,
            loss_cls=dict(type='CrossEntropyLoss'),
            spatial_type='avg',
            consensus=dict(type='AvgConsensus', dim=1),
            dropout_ratio=0.5,
            init_std=0.001,
            is_shift=True,
            temporal_pool=False,
        )

    def forward(self, x):
        x = einops.rearrange(x, 'N S C H W -> (N S) C H W')
        o = self.backbone(x)
        return self.head(o, num_segs=1)
        # return self.tsm(x)


class LitModel(pl.LightningModule):

    def __init__(self, backbone_model, learning_rate):
        super().__init__()
        self.example_input_array = torch.randn(1, 8, 3, 224, 224)
        self.save_hyperparameters()
        self.model = VideoModel(backbone_model=backbone_model,
                                num_classes=2,
                                num_segments=8)
        self.loss_module = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_module(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("train/acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_module(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("val/acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val/loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_module(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("test/acc", acc, on_step=False, on_epoch=True)
        self.log('test/loss', loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        # optimizer = optim.SGD(self.parameters(),
        #                       lr=self.learning_rate,
        #                       momentum=0.9,
        #                       weight_decay=0.0001)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train/loss",
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)


def export_model(ckpt):
    model = LitModel.load_from_checkpoint(ckpt)
    model.eval()
    onnx_path = ckpt.replace('.ckpt', '.onnx')
    model.to_onnx(onnx_path, export_params=True)


def main(args):
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    backbone = args.backbone
    action = args.action

    pl.seed_everything(0)
    # callbacks
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    early_stop = pl.callbacks.early_stopping.EarlyStopping(monitor='train/loss',
                                                           mode='min',
                                                           patience=10)
    model = LitModel(backbone, lr)
    # loggers
    PROJECT = 'binary_video_model'
    NAME = f'{action}_{backbone}'
    tensorboard_logger = pl.loggers.TensorBoardLogger(save_dir=os.path.join(
        proj_root, f'lightning_logs/tensorboard/{PROJECT}'),
                                                      name=NAME,
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
    loggers = [tensorboard_logger]
    if args.wandb:
        wandb_logger = pl.loggers.WandbLogger(save_dir=os.path.join(
            proj_root, 'lightning_logs'),
                                              project=PROJECT,
                                              name=NAME)
        wandb_logger.log_hyperparams(args)
        wandb_logger.watch(model, log="all")

        loggers.append(wandb_logger)

    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    trainer = pl.Trainer(
        default_root_dir=os.path.join(proj_root, 'lightning_logs'),
        max_epochs=epochs,
        accelerator='gpu',
        devices=1,
        logger=loggers,
        callbacks=[lr_monitor, early_stop],
        auto_lr_find=True,
    )
    train_loader, val_loader, test_loader = get_data_loaders(action, batch_size)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', action='store_true', help='train or export')
    parser.add_argument('-ck',
                        '--ckpt',
                        type=str,
                        default=None,
                        help='checkpoint to load')
    parser.add_argument('-lr', '--lr', type=float, default=5e-6, help='learning rate')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='epochs')
    parser.add_argument('-b',
                        '--backbone',
                        type=str,
                        default='resnet50',
                        help='backbone of the TSM model')
    parser.add_argument('-a', '--action', type=str, default='squat', help='action')
    parser.add_argument('-bs', '--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--wandb', action='store_true', help='add logger wandb')
    args = parser.parse_args()
    if args.train:
        main(args)
    elif args.ckpt:
        export_model(args.ckpt)