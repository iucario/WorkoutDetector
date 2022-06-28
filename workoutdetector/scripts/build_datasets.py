import argparse
import os
from os.path import join as osj
import numpy as np
import pandas as pd
from PIL import Image
from torchvision.io.video import read_video
from workoutdetector.settings import PROJ_ROOT
from subprocess import call


def build_image_rep() -> None:
    """Creates images for torchvision.datasets.ImageFolder. Repetition states.

    Args:
        data_dir: path like `data/RepCount/video`
        anno_path: csv file path
        dest_dir: images will be saved in `dest_dir/train/category/image_1.png`.
    """
    data_dir = osj(PROJ_ROOT, 'data/RepCount/video')
    anno_path = osj(PROJ_ROOT, '/data/RepCount/annotation.csv')
    dest_dir = osj(PROJ_ROOT, 'data/RepCount/rep_image')
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


def build_workout():
    """Creates the Workout dataset in `data/Workout/`.
    Expects dataset Countix and RepCount to already in `data/`.
    Merge the two datasets and relabel the classes.
    9 classes from RepCount and 10 from Countix. To a new dataset Workout with 11 classes.
    """

    assert os.path.isdir(os.path.join(PROJ_ROOT, 'data/RepCount'))
    assert os.path.isdir(os.path.join(PROJ_ROOT, 'data/Countix'))

    classes = [
        'front_raise', 'pull_up', 'squat', 'bench_pressing', 'jumping_jack', 'situp',
        'push_up', 'battle_rope', 'exercising_arm', 'lunge', 'mountain_climber'
    ]

    repcount_class = []
    countix_class = []

    with open(os.path.join(PROJ_ROOT, 'datasets/RepCount/classes.txt')) as f:
        for line in f:
            repcount_class.append(line.strip())

    with open(os.path.join(PROJ_ROOT, 'datasets/Countix/classes.txt')) as f:
        for line in f:
            countix_class.append(line.strip())

    # map RepCount classes to Workout classes
    repcount_to = [
        'front_raise', 'pull_up', 'squat', 'bench_pressing', 'jumping_jack', 'situp',
        'push_up', 'battle_rope'
    ]
    repcount_d = {}
    for i, x in enumerate(repcount_class):
        if x in repcount_to:
            repcount_d[i] = classes.index(x)

    # map Countix classes to Workout classes
    countix_to = [
        'exercising_arm', 'bench_pressing', 'front_raise', 'squat', 'jumping_jack',
        'lunge', 'mountain_climber', 'pull_up', 'push_up', 'situp'
    ]

    countix_d = dict()
    for i, x in enumerate(countix_class):
        y = countix_to[i]
        j = classes.index(y)
        countix_d[i] = j

    print(repcount_d)
    print(countix_d)

    # write the new labels to file

    def build(label_map: dict, prefix: str, input_txt: str, output_txt: str) -> None:
        set_type = input_txt.split('/')[-1].split('.')[0]
        with open(input_txt, 'r') as f:
            lines = f.readlines()
        with open(output_txt, 'w') as f:
            for line in lines:
                path, length, label = line.rstrip().split()
                path, length, label = line.split()
                i = int(label)
                if i in label_map:
                    label = label_map[i]
                else:
                    continue
                path = '/'.join([prefix, set_type, path])
                f.write(f'{path} {length} {label}\n')

    # new large dataset will be in path: data/Workout
    build(repcount_d, 'RepCount',
          os.path.join(PROJ_ROOT, 'data/RepCount/rawframes/train.txt'),
          os.path.join(PROJ_ROOT, 'data/Workout/rawframes/train_repcount.txt'))
    build(repcount_d, 'RepCount',
          os.path.join(PROJ_ROOT, 'data/RepCount/rawframes/val.txt'),
          os.path.join(PROJ_ROOT, 'data/Workout/rawframes/val_repcount.txt'))
    build(repcount_d, 'RepCount',
          os.path.join(PROJ_ROOT, 'data/RepCount/rawframes/test.txt'),
          os.path.join(PROJ_ROOT, 'data/Workout/rawframes/test_repcount.txt'))
    # Countix
    build(countix_d, 'Countix', os.path.join(PROJ_ROOT,
                                             'data/Countix/rawframes/train.txt'),
          os.path.join(PROJ_ROOT, 'data/Workout/rawframes/train_countix.txt'))
    build(countix_d, 'Countix', os.path.join(PROJ_ROOT, 'data/Countix/rawframes/val.txt'),
          os.path.join(PROJ_ROOT, 'data/Workout/rawframes/val_countix.txt'))

    # Merge files
    # cat train_repcount.txt train_countix.txt > train.txt
    # cat val_repcount.txt val_countix.txt > val.txt
    # cat test_repcount.txt > test.txt

    # link to Workouts
    # ln -s data/RepCount/rawframes/train data/Workout/rawframes/RepCount/train
    # ln -s data/RepCount/rawframes/val data/Workout/rawframes/RepCount/val
    # ln -s data/RepCount/rawframes/test data/Workout/rawframes/RepCount/test
    # ln -s data/Countix/rawframes/train data/Workout/rawframes/Countix/train
    # ln -s data/Countix/rawframes/val data/Workout/rawframes/Countix/val


def parse_args(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('func',
                        type=str,
                        help='function to run',
                        choices=['build_workout', 'build_video'])
    args = parser.parse_args(argv)
    func = globals()[args.func]
    func()


if __name__ == '__main__':
    parse_args()
