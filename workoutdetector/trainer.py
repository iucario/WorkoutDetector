import argparse
import os
import os.path as osp
import time
from os.path import join as osj
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import timm
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import yaml
from fvcore.common.config import CfgNode
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import (LearningRateMonitor, ModelCheckpoint,
                                         early_stopping)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from workoutdetector.datasets import ImageDataset
from workoutdetector.settings import PROJ_ROOT


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train image model')
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default=osj(PROJ_ROOT, "workoutdetector/configs/lit_img.yaml"),
        type=str,
    )
    parser.add_argument(
        "opts",
        help="See workoutdetector/configs/lit_img.yaml for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args(argv)


def load_config(args) -> CfgNode:
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `cfg_file`.
    """
    cfg = CfgNode(new_allowed=True)
    cfg.merge_from_file(args.cfg_file)
    if args.opts is not None:
        cfg.merge_from_list(args.opts)
    return cfg


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


class Detector():

    def __init__(self):
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        self.model.cuda()

    @torch.no_grad()
    def detect(self, images: List[Tensor], threshold: float = 0.7) -> List[Tensor]:
        """Detect human

        Args:
            images, List[Tensor]: list of images to fastrcnn
            threshold, float: threshold. Defaults to 0.7.
        Returns:
            List[Tensor]: list of bounding boxes of shape (N, 4)
        """

        results: List[Dict[str, Tensor]] = self.model(images)
        persons: List[Tensor] = []
        for r in results:
            persons.append(self._get_one_frame(r, threshold))
        return persons

    def _get_one_frame(self, result: dict, threshold: float = 0.7) -> Tensor:
        human_inds = result['labels'] == 1
        person_boxes = result['boxes'][human_inds]
        scores = result['scores'][human_inds]
        person_boxes = person_boxes[scores > threshold]
        person_boxes.cpu().numpy()
        return person_boxes

    def crop_person(self, images: List[Tensor]) -> List[Tensor]:
        """Crop person from images. One person per image.

        Args:
            image, List[Tensor]: image to crop person from.
        Returns:
            List[Tensor]: cropped image padded or resized to 224x224.
        TODO:
            Person tracking in video
        """

        boxes = self.detect(images)
        cropped_images: List[Tensor] = []
        for persons in boxes:
            x1, y1, x2, y2 = persons[0]
            # make box larger by 10%
            x1, y1, x2, y2 = list(map(int, [x1 * 0.9, y1 * 0.9, x2 * 1.1, y2 * 1.1]))
            person = TF.crop(images[0], x1, y1, x2 - x1, y2 - y1)
            h, w = person.shape[:2]
            max_hw = max(h, w)
            person = TF.pad(person,
                            padding=(max_hw - h, max_hw - w),
                            fill=0,
                            padding_mode='constant')
            person = TF.resize(person, size=(224, 224))
            cropped_images.append(person)
            assert person.shape[-2:] == (224, 224), f"{person.shape}"
        return cropped_images


class MyModel(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(),
                                    nn.Linear(128, 1))

    def forward(self, x):
        return self.layers(x)


class LitImageModel(LightningModule):

    def __init__(self, cfg: CfgNode):
        super().__init__()
        self.example_input_array = torch.randn(1, 3, 224, 224)
        self.save_hyperparameters()
        backbone = timm.create_model(cfg.model.backbone_model,
                                     pretrained=True,
                                     num_classes=cfg.model.num_class)
        self.classifier = backbone
        self.loss_module = nn.CrossEntropyLoss()
        self.cfg = cfg

    def forward(self, x):
        return self.classifier(x)

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

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def configure_optimizers(self):
        cfg = self.cfg
        OPTIMIZER = cfg.optimizer.method.lower()
        SCHEDULER = cfg.lr_scheduler.policy.lower()
        if OPTIMIZER == 'sgd':
            optimizer = optim.SGD(self.parameters(),
                                  lr=cfg.optimizer.lr,
                                  momentum=cfg.optimizer.momentum,
                                  weight_decay=cfg.optimizer.weight_decay)
        elif OPTIMIZER == 'adamw':
            optimizer = optim.AdamW(self.parameters(),
                                    lr=cfg.optimizer.lr,
                                    eps=cfg.optimizer.eps,
                                    weight_decay=cfg.optimizer.weight_decay)
        else:
            raise NotImplementedError(
                f'Not implemented optimizer: {cfg.optimizer.method}')
        if SCHEDULER == 'steplr':
            scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                  step_size=cfg.lr_scheduler.step,
                                                  gamma=cfg.lr_scheduler.gamma)
        else:
            raise NotImplementedError(f'Not implemented lr schedular: {cfg.lr_schedular}')
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val/loss",
        }


class ImageDataModule(LightningDataModule):
    """General image dataset
    label text files of [image.png class] are required

    Args:
        cfg (CfgNode): configs of cfg.data
        transform: Optional
        target_transform: Optional
        is_train: bool, train or test. Default True
    """

    def __init__(self,
                 cfg: CfgNode,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 is_train: bool = True) -> None:
        super().__init__()
        if is_train:
            assert osp.exists(osj(
                cfg.data_root,
                cfg.train.anno)), f'{osj(cfg.data_root, cfg.train.anno)} does not exist'
            assert osp.exists(
                osj(cfg.data_root,
                    cfg.val.anno)), f'{osj(cfg.data_root, cfg.val.anno)} does not exist'
        if cfg.test.anno:
            assert osp.exists(
                osj(cfg.data_root,
                    cfg.test.anno)), f'{osj(cfg.data_root, cfg.test.anno)} does not exist'
        self.cfg = cfg
        self.transform = transform
        self.target_transform = target_transform

    def train_dataloader(self):
        prefix = '' if self.cfg.train.data_prefix is None else self.cfg.train.data_prefix
        train_set = ImageDataset(self.cfg.data_root, prefix, self.cfg.train.anno,
                                 self.transform)
        loader = DataLoader(train_set,
                            num_workers=self.cfg.num_workers,
                            batch_size=self.cfg.batch_size,
                            shuffle=True)
        return loader

    def val_dataloader(self):
        prefix = '' if self.cfg.train.data_prefix is None else self.cfg.train.data_prefix
        val_set = ImageDataset(self.cfg.data_root, prefix, self.cfg.val.anno,
                               self.transform)
        loader = DataLoader(val_set,
                            num_workers=self.cfg.num_workers,
                            batch_size=self.cfg.batch_size,
                            shuffle=False)
        return loader

    def test_dataloader(self):
        prefix = '' if self.cfg.train.data_prefix is None else self.cfg.train.data_prefix
        if self.cfg.test.anno:
            test_set = ImageDataset(self.cfg.data_root, prefix, self.cfg.test.anno,
                                    self.transform)
            loader = DataLoader(test_set,
                                num_workers=self.cfg.num_workers,
                                batch_size=self.cfg.batch_size,
                                shuffle=False)
            return loader
        else:
            return self.val_dataloader()


def export_model(ckpt: str, onnx_path: Optional[str] = None) -> None:
    model = LitImageModel.load_from_checkpoint(ckpt)
    model.eval()
    if onnx_path is None:
        onnx_path = ckpt.replace('.ckpt', '.onnx')
    model.to_onnx(onnx_path, input_sample=model.example_input_array, export_params=True)


def cfg_to_dict(cfg: CfgNode) -> dict:
    x = cfg.dump()
    y = yaml.safe_load(x)
    return y


def test(cfg: CfgNode) -> None:
    data_module = ImageDataModule(cfg.data,
                                  transform=data_transforms['train'],
                                  target_transform=data_transforms['test'],
                                  is_train=False)
    model = LitImageModel.load_from_checkpoint(cfg.checkpoint)
    trainer = Trainer(
        default_root_dir=cfg.trainer.default_root_dir,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
    )
    trainer.test(model, data_module)


def train(cfg: CfgNode) -> None:
    data_module = ImageDataModule(cfg.data,
                                  transform=data_transforms['train'],
                                  target_transform=data_transforms['test'])
    model = LitImageModel(cfg)

    timenow = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    # callbacks
    CALLBACKS: List[Any] = []

    lr_monitor = LearningRateMonitor(logging_interval='step')
    CALLBACKS.append(lr_monitor)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        save_weights_only=True,
        monitor="val/acc",
        mode="max",
        dirpath=osj(cfg.trainer.default_root_dir, 'checkpoints'),
        filename="best-val-acc={val/acc:.2f}-epoch={epoch:02d}" + f"-{timenow}",
        auto_insert_metric_name=False)
    CALLBACKS.append(checkpoint_callback)

    if cfg.trainer.early_stopping:
        early_stop = early_stopping.EarlyStopping(monitor='train/loss',
                                                  mode='min',
                                                  patience=cfg.trainer.patience)
        CALLBACKS.append(early_stop)

    # loggers
    cfg_dict = cfg_to_dict(cfg)
    LOGGER: List[Any] = []
    if cfg.log.enable_wandb:
        wandb_logger = WandbLogger(
            save_dir=osj(cfg.log.output_dir),
            project=cfg.log.wandb.project,
            name=cfg.log.name,
            offline=cfg.log.wandb.offline,
        )
        wandb_logger.log_hyperparams(cfg_dict)
        wandb_logger.watch(model, log="all")
        LOGGER.append(wandb_logger)

    if cfg.log.enable_tensorboard:
        tensorboard_logger = TensorBoardLogger(save_dir=cfg.log.output_dir,
                                               name=cfg.log.name,
                                               default_hp_metric=False)
        tensorboard_logger.log_hyperparams(cfg_dict,
                                           metrics={
                                               'train/acc': 0,
                                               'val/acc': 0,
                                               'test/acc': 0,
                                               'train/loss': 1,
                                               'val/loss': 1,
                                               'test/loss': 1
                                           })
        LOGGER.append(tensorboard_logger)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    trainer = Trainer(
        default_root_dir=cfg.trainer.default_root_dir,
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        logger=LOGGER,
        callbacks=CALLBACKS,
        auto_lr_find=cfg.trainer.auto_lr_find,
        log_every_n_steps=cfg.log.log_every_n_steps,
    )

    trainer.fit(model, data_module)

    model.load_from_checkpoint(checkpoint_callback.best_model_path)
    print(checkpoint_callback.best_model_path)
    wandb_logger.experiment.config[
        'best_model_path'] = checkpoint_callback.best_model_path
    
    trainer.test(model, data_module)


def main(cfg: CfgNode) -> None:
    pl.seed_everything(cfg.seed)

    if cfg.train:
        train(cfg)
    else:
        test(cfg)


if __name__ == '__main__':

    args = parse_args()
    cfg = load_config(args)
    main(cfg)
