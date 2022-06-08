import math
from WorkoutDetector.utils.common import Repcount
import random
import sys
import os
import PIL
import cv2
import pandas as pd
import numpy as np
import yaml
import torch
from matplotlib import pyplot as plt
from torch import Tensor, nn
from torchvision.io import read_image
import torchvision.transforms as T
import einops

config = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), 'config.yml')))

CLASSES = [
    'front_raise', 'pull_up', 'squat', 'bench_pressing', 'jumping_jack', 'situp',
    'push_up', 'battle_rope', 'exercising_arm', 'lunge', 'mountain_climber'
]


class ImageDataset(torch.utils.data.Dataset):
    """Binary class image dataset from Repcount dataset. Start state is 0, mid state is 1.
    Number of 2*counts images are returned for each video.

    Args:
        classname: str, action class name.
        split: str, train or val or test.
        transform: torchvision.transforms.Compose, transform to apply to image.
    """

    def __init__(self, classname='squat', split='train', transform=None):
        self.classname = classname
        self.transform = transform
        repcount = Repcount()
        df = repcount.get_anno(split)
        df = df[df['type'] == classname]
        self.data_root = os.path.join(repcount.data_root, 'rawframes', split)
        vids = df['name'].values
        images = []
        labels = []
        for vid in vids:
            vid = vid.split('.')[0]
            count, reps = repcount.get_count(vid)
            for start, end in zip(reps[0::2], reps[1::2]):
                start += 1
                end += 1
                mid = (start + end) // 2
                images.append(f'{vid}/img_{start:05}.jpg')
                images.append(f'{vid}/img_{mid:05}.jpg')
                labels.append(0)
                labels.append(1)
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        image = self.images[index]
        img_path = os.path.join(self.data_root, image)
        image = read_image(img_path)
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.images)


def sample_frames(total: int, num: int, offset=0):
    """
    Uniformly sample num frames from video
    Input: total frames, number of frames to sample
    Returns: list of frame indices starting from offset
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
        classname: str, action class name.
        split: str, train or val or test.
        num_image: int, number of images in a super image.
        transform: torchvision.transforms.Compose, transform to apply to image.
    """

    def __init__(self, classname='squat', split='train', num_image=9, transform=None):
        self.classname = classname
        self.transform = transform
        repcount = Repcount()
        df = repcount.get_anno(split)
        df = df[df['type'] == classname]
        data_root = os.path.join(repcount.data_root, 'rawframes', split)
        vids = df['name'].values
        images = []
        labels = []
        for vid in vids:
            vid = vid.split('.')[0]
            count, reps = repcount.get_count(vid)
            for start, end in zip(reps[0::2], reps[1::2]):
                start += 1
                end += 1
                mid = (start + end) // 2
                samples_start = sample_frames(mid - start, num_image, start)
                samples_end = sample_frames(end - mid, num_image, mid)
                # concat 9 image into 1 super image
                imgs_start = []
                for i in samples_start:
                    imgs_start.append(os.path.join(data_root, f'{vid}/img_{i:05}.jpg'))
                imgs_end = []
                for i in samples_end:
                    imgs_end.append(os.path.join(data_root, f'{vid}/img_{i:05}.jpg'))
                images.append(imgs_start)
                images.append(imgs_end)
                labels.append(0)
                labels.append(1)
        self.images = images
        self.labels = labels
        self.num_image = num_image

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


if __name__ == '__main__':
    dataset = SuperImageDataset(classname='squat',
                                split='train',
                                num_image=9,
                                transform=None)
    print(len(dataset))
    plt.imshow(dataset[np.random.randint(len(dataset))][0].permute(1, 2, 0))
    plt.show()
