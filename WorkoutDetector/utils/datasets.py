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

config = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), 'config.yml')))

CLASSES = ['front_raise', 'pull_up', 'squat', 'bench_pressing', 'jumping_jack', 'situp',
           'push_up', 'battle_rope', 'exercising_arm', 'lunge', 'mountain_climber']

class ImageDataset(torch.utils.data.Dataset):
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


if __name__ == '__main__':
    dataset = ImageDataset(classname='squat', split='train')
    print(len(dataset))
    print(dataset[0].shape)
