import os

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.io.video import read_video
from workoutdetector.settings import PROJ_ROOT


def build_image_rep(data_dir: str, anno_path: str, dest_dir: str) -> None:
    """Creates images for torchvision.datasets.ImageFolder. Repetition states.

    Args:
        data_dir: path like `data/RepCount/video`
        anno_path: csv file path
        dest_dir: images will be saved in `dest_dir/train/category/image_1.png`.
    """

    CLASSES = ['situp', 'push_up', 'pull_up', 'jump_jack', 'squat', 'front_raise']
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(dest_dir, split), exist_ok=True)
        for i in range(len(CLASSES) * 2):
            os.makedirs(os.path.join(dest_dir, split, str(i)), exist_ok=True)
    train_csv = open(os.path.join(dest_dir, 'train.csv'), 'a')
    val_csv = open(os.path.join(dest_dir, 'val.csv'), 'a')
    test_csv = open(os.path.join(dest_dir, 'test.csv'), 'a')

    anno = pd.read_csv(anno_path)
    for i, row in anno.iterrows():

        if row['class_'] not in CLASSES:
            continue
        count = int(row['count'])
        if count == 0:
            continue

        split = row["split"]
        video_path = os.path.join(data_dir, split, row['name'])
        video_name = row['name']

        reps = [int(x) for x in row['reps'].split()]
        video = read_video(video_path)[0]
        start_idx = reps[0]
        end_idx = reps[1]
        mid_idx = (start_idx + end_idx) // 2

        name_no_ext = video_name.split('.')[0]
        rep_class = CLASSES.index(row["class_"]) * 2
        print(video_path, start_idx, mid_idx)
        Image.fromarray(video[end_idx].numpy()).save(
            os.path.join(dest_dir, split, str(rep_class), f'{name_no_ext}.png'))
        Image.fromarray(video[mid_idx].numpy()).save(
            os.path.join(dest_dir, split, str(rep_class + 1), f'{name_no_ext}.png'))

    print('Done')


if __name__ == '__main__':
    data_dir = os.path.expanduser('~/data/RepCount/video')
    anno_path = os.path.expanduser('~/data/RepCount/annotation.csv')
    dest_dir = os.path.expanduser('~/data/RepCount/rep_image')
    build_image_rep(data_dir, anno_path, dest_dir)
