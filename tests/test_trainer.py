from workoutdetector.settings import PROJ_ROOT, REPCOUNT_ANNO_PATH
import torch
from torchvision.io import read_image
import torchvision.transforms.functional as TF
import os
import os.path as osp
from os.path import join as osj
from fvcore.common.config import CfgNode
from workoutdetector.trainer import DataModule


def test_config():
    