import os
from os.path import join as osj

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import read_image
from workoutdetector.datasets.transform import Detector, PersonCrop
from workoutdetector.settings import PROJ_ROOT


def test_Detector():
    """Test object detection"""

    det = Detector()
    img = torch.rand(2, 3, 224, 224)
    y = det.detect(img.cuda())
    assert len(y) == 2, 'detect must return 2 tensors'

    img_human = read_image(osj(PROJ_ROOT, 'tests/data/test.png')).float().cuda()
    y = det.detect([img_human])
    assert y[0].shape == (1, 4), 'should detect one person'

    cropped = det.crop_person([img_human])
    assert cropped[0] is not None, 'should crop person'
    assert cropped[0].shape[-2:] == (224, 224), 'should crop to 224x224'


def test_PersonCrop():

    func = PersonCrop()
    img = torch.rand(2, 3, 600, 400).cuda()
    try:
        y = func(img)
    except Exception as e:
        pytest.fail(e)
    assert y.shape[:-2] == img.shape[:-2]
