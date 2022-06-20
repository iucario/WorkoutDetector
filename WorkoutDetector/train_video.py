from WorkoutDetector.settings import PROJ_ROOT
import argparse
from typing import Tuple
from WorkoutDetector.datasets import RepcountVideoDataset, RepcountRecognitionDataset
import torch
from torch.utils.data import DataLoader
import os
from torch import optim, nn, utils, Tensor
import pytorch_lightning as pl
import torchvision.transforms as T
import timm
import einops
import time

import pytorchvideo.data
from pytorchvideo.models import create_resnet, create_slowfast
from pytorchvideo.models.x3d import create_x3d

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
                     batch_size: int,
                     num_segments: int = 8) -> Tuple[DataLoader, DataLoader, DataLoader]:
    data_root = os.path.join(PROJ_ROOT, 'data')
    train_set = RepcountVideoDataset(root=data_root,
                                     action=action,
                                     split='train',
                                     num_segments=num_segments,
                                     transform=data_transforms['train'])
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)
    val_set = RepcountVideoDataset(root=data_root,
                                   action=action,
                                   split='val',
                                   num_segments=num_segments,
                                   transform=data_transforms['val'])
    val_loader = DataLoader(val_set,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True)
    test_set = RepcountVideoDataset(root=data_root,
                                    action=action,
                                    split='test',
                                    num_segments=num_segments,
                                    transform=data_transforms['val'])
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True)
    return train_loader, val_loader, test_loader


class DebugModel(nn.Module):
    """Simple CNN model for debugging."""

    def __init__(self, num_class: int):
        super(DebugModel, self).__init__()
        self.num_class = num_class

    def forward(self, x):
        return x


class VideoModel(nn.Module):

    def __init__(self,
                 model_name: str,
                 model_num_class: int = 2,
                 num_segments: int = 8,
                 pretrained: bool = False):
        super(VideoModel, self).__init__()
        # slowfast_model = pytorchvideo.models.create_slowfast(
        #     model_num_class=model_num_class, input_channels=input_channels)
        # self.model = slowfast_model
        assert model_name in ['x3d_s', 'x3d_m', 'slow_r50']
        if model_name == 'x3d_s':
            model = create_x3d(model_num_class=model_num_class,
                               input_clip_length=num_segments,
                               input_crop_size=224)
            to_remove = ['blocks.5.proj.weight', 'blocks.5.proj.bias']
            if pretrained:
                ckpt = torch.load(
                    os.path.expanduser('~/.cache/torch/hub/checkpoints/X3D_S.pyth'))
                state_dict = ckpt["model_state"]
                for key in list(state_dict.keys()):
                    if key in to_remove:
                        state_dict.pop(key)
                model.load_state_dict(state_dict)
                in_features = list(model.modules())[-3].in_features
        else:
            model = create_resnet(
                stem_conv_kernel_size=(1, 7, 7),
                head_pool_kernel_size=(8, 7, 7),
                model_depth=50,
            )
        self.model = model

    def forward(self, x):
        x = self.model(x)
        return x


class LitModel(pl.LightningModule):

    def __init__(self, model_name, num_class: int, num_segments: int,
                 learning_rate: float):
        super().__init__()
        self.example_input_array = torch.randn(1, 3, num_segments, 224, 224)

        self.save_hyperparameters()
        self.model = VideoModel(model_name,
                                model_num_class=num_class,
                                num_segments=num_segments)
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
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
        #                                                        10,
        #                                                        last_epoch=-1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train/loss",
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)


def debug():
    """Simple action recognition to verify network works"""

    from WorkoutDetector.datasets import DebugDataset

    data_root = os.path.join(PROJ_ROOT, 'data/RepCount')
    ACTIONS = ['situp', 'push_up', 'pull_up', 'squat', 'jump_jack']
    NUM_CLASS = len(ACTIONS)
    NUM_SEGMENTS = 8
    LR = 1e-5
    MODEL_NAME = 'x3d_s'
    BATCH_SIZE = 2
    EPOCHS = 20

    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(0)

    trainset = RepcountRecognitionDataset(data_root, 'train', ACTIONS, NUM_SEGMENTS,
                                          data_transforms['train'])
    valset = RepcountRecognitionDataset(data_root, 'val', ACTIONS, NUM_SEGMENTS,
                                        data_transforms['val'])
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False)

    loggers = [
        pl.loggers.TensorBoardLogger(save_dir=os.path.join(PROJ_ROOT, 'log/tensorboard'),
                                     log_graph=True,
                                     name='debug'),
    ]

    # callbacks
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step',
                                                  log_momentum=True)
    early_stop = pl.callbacks.early_stopping.EarlyStopping(monitor='train/loss',
                                                           mode='min',
                                                           patience=20)
    model = LitModel(model_name=MODEL_NAME,
                     num_class=NUM_CLASS,
                     num_segments=NUM_SEGMENTS,
                     learning_rate=LR)

    trainer = pl.Trainer(
        default_root_dir=os.path.join(PROJ_ROOT, 'log/lightning_logs'),
        max_epochs=EPOCHS,
        accelerator='gpu',
        logger=loggers,
        devices=1,
        callbacks=[lr_monitor, early_stop],
        overfit_batches=10,
        detect_anomaly=True,
        auto_lr_find=True,
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, val_loader)


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
    model = LitModel(model_name=args.model,
                     num_class=2,
                     num_segments=args.seg,
                     learning_rate=lr)
    # loggers
    PROJECT = 'binary_video_model'
    NAME = f'{action}_{backbone}'
    tensorboard_logger = pl.loggers.TensorBoardLogger(save_dir=os.path.join(
        PROJ_ROOT, f'log/tensorboard/{PROJECT}'),
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
        wandb_logger = pl.loggers.WandbLogger(save_dir=os.path.join(PROJ_ROOT, 'log/'),
                                              project=PROJECT,
                                              name=NAME)
        wandb_logger.log_hyperparams(args)
        wandb_logger.watch(model, log="all")

        loggers.append(wandb_logger)

    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    trainer = pl.Trainer(
        default_root_dir=os.path.join(PROJ_ROOT, 'lightning_logs'),
        max_epochs=epochs,
        accelerator='gpu',
        devices=1,
        logger=loggers,
        callbacks=[lr_monitor, early_stop],
        auto_lr_find=True,
    )
    train_loader, val_loader, test_loader = get_data_loaders(action,
                                                             batch_size,
                                                             num_segments=args.seg)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', action='store_true', help='train or export')
    parser.add_argument('-ck',
                        '--ckpt',
                        type=str,
                        default=None,
                        help='checkpoint to load')
    parser.add_argument('-lr', '--lr', type=float, default=5e-6, help='learning rate')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='epochs')
    parser.add_argument('-m', '--model', type=str, default='x3d_s', help='model name')
    parser.add_argument('-b',
                        '--backbone',
                        type=str,
                        default='resnet50',
                        help='backbone of the TSM model')
    parser.add_argument('-a', '--action', type=str, default='squat', help='action')
    parser.add_argument('-bs', '--batch-size', type=int, default=2, help='batch size')
    parser.add_argument('--seg', type=int, default=8, help='number of segments')
    parser.add_argument('--wandb', action='store_true', help='add logger wandb')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    args = parser.parse_args(argv)

    return args


if __name__ == '__main__':
    args_train = [
        '-t', '-a', 'jump_jack', '-e', '5', '-lr', '1e-3', '--seg', '16', '--model',
        'x3d_m'
    ]
    args = parse_args()
    if args.train:
        main(args)
    elif args.debug:
        debug()
    elif args.ckpt:
        export_model(args.ckpt)
