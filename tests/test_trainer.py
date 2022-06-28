from workoutdetector.trainer import Detector
from workoutdetector.settings import PROJ_ROOT, REPCOUNT_ANNO_PATH
import torch
from torchvision.io import read_image
import torchvision.transforms.functional as TF
import os
import os.path as osp
from os.path import join as osj


def test_Detector():
    """Test object detection"""
    det = Detector()
    img = torch.rand(2, 3, 224, 224)
    y = det.detect(img.cuda())
    assert len(y) == 2, 'detect must return 2 tensors'

    img_human = read_image(osj(PROJ_ROOT, 'images/test.png')).float().cuda()
    y = det.detect([img_human])
    assert y[0].shape == (1, 4), 'should detect one person'

    cropped = det.crop_person([img_human])
    assert cropped[0] is not None, 'should crop person'
    assert cropped[0].shape[-2:] == (224, 224), 'should crop to 224x224'