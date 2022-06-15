import math
import os
import pandas as pd
import numpy as np
import yaml
import torch
from matplotlib import pyplot as plt
from torch import Tensor, nn
from torchvision.io import read_image
import torchvision.transforms as T
import einops


CLASSES = [
    'front_raise', 'pull_up', 'squat', 'bench_pressing', 'jumping_jack', 'situp',
    'push_up', 'battle_rope', 'exercising_arm', 'lunge', 'mountain_climber'
]


def sample_frames(total: int, num: int, offset=0):
    """Uniformly sample num frames from video
    
    Args:
        total: int, total frames, 
        num: int, number of frames to sample
        offset: int, offset from start of video
    Returns: 
        list of frame indices starting from offset
    """

    if total < num:
        # repeat frames if total < num
        repeats = math.ceil(num / total)
        data = [x for x in range(total) for _ in range(repeats)]
        total = len(data)
    else:
        data = list(range(total))
    interval = total // num
    indices = np.arange(0, total, interval)[:num]
    for i, x in enumerate(indices):
        rand = np.random.randint(0, interval)
        if i == num - 1:
            upper = total
            rand = np.random.randint(0, upper - x)
        else:
            upper = min(interval * (i + 1), total)
        indices[i] = (x + rand) % upper
    assert len(indices) == num, f'len(indices)={len(indices)}'
    for i in range(1, len(indices)):
        assert indices[i] > indices[i - 1], f'indices[{i}]={indices[i]}'
    return [data[i] + offset for i in indices]


class SuperImageDataset(torch.utils.data.Dataset):
    """Rearrange the images to make a super image.
    Fan, Q. and Panda, R. 2022. CAN AN IMAGE CLASSIFIER SUFFICE FOR ACTION RECOGNITION?
    https://openreview.net/pdf?id=qhkFX-HLuHV

    Args:
        images: list of images
        classname: str, action class name.
        split: str, train or val or test.
        num_image: int, number of images in a super image.
        transform: torchvision.transforms.Compose, transform to apply to image.
    """

    def __init__(self, images: list, labels: list, num_image=9, transform=None):
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        images = [read_image(img) for img in self.images[index]]
        label = self.labels[index]
        if self.transform:
            images = [self.transform(img) for img in images]
        super_image = torch.stack(images, dim=1)
        if self.num_image == 9:
            super_image = einops.rearrange(super_image,
                                           'c (sh sw) h w -> c (sh h) (sw w)',
                                           sh=3,
                                           sw=3)
        elif self.num_image == 4:
            super_image = einops.rearrange(super_image,
                                           'c (sh sw) h w -> c (sh h) (sw w)',
                                           sh=2,
                                           sw=2)
        else:
            raise ValueError(f'num_image={self.num_image}. Only support 4 or 9')
        super_image = T.Resize((224, 224))(super_image)
        return super_image, label

    def __len__(self):
        return len(self.labels)
