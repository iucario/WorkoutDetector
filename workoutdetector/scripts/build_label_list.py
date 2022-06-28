import os
import os.path as osp
from os.path import join as osj

import pandas as pd
import yaml
from torchvision.io import VideoReader
from workoutdetector.datasets import RepcountDataset
from workoutdetector.utils.inference_count import PROJ_ROOT


def build_label() -> None:
    train = os.path.join(PROJ_ROOT, 'data/Countix/rawframes/train.txt')
    val = os.path.join('data/Countix/rawframes/val.txt')

    traindir = os.path.join(PROJ_ROOT, 'data/Workouts/rawframes/Countix/train')
    valdir = os.path.join(PROJ_ROOT, 'data/Workouts/rawframes/Countix/val')

    traindf = pd.read_csv(os.path.join(PROJ_ROOT, 'datasets/Countix/workouts_train.csv'))
    valdf = pd.read_csv(os.path.join(PROJ_ROOT, 'datasets/Countix/workouts_val.csv'))
    classes = []
    with open(os.path.join(PROJ_ROOT, 'datasets/Countix/classes.txt')) as f:
        classes = [line.rstrip() for line in f]

    with open(train, 'w') as f:
        for i, row in traindf.iterrows():
            vid = row['video_id']
            label = classes.index(row['class'])
            if os.path.exists(os.path.join(traindir, vid)):
                num_frames = len(os.listdir(os.path.join(traindir, vid)))
                f.write(f'{vid} {num_frames} {label}\n')

    with open(val, 'w') as f:
        for i, row in valdf.iterrows():
            vid = row['video_id']
            label = classes.index(row['class'])
            if os.path.exists(os.path.join(valdir, vid)):
                num_frames = len(os.listdir(os.path.join(valdir, vid)))
                f.write(f'{vid} {num_frames} {label}\n')


def build_with_start(data_root: str, dst_dir: str) -> None:
    """Creates label files for video classification.

    Args:
        data_root: Directory to RepCount dataset. Same as in RepCountDataset.
        dst_dir: Path to dir where txt files will be created.

    Note:
        For each action and `all-{split}.txt` file is created.
        Lines in the label files are in the format:
            video_path, start, length, label
    """

    # pop bench_pressing because there are too many errors in the annotation
    CLASSES = ['situp', 'push_up', 'pull_up', 'jump_jack', 'squat', 'front_raise']
    MAX_REPS = 2  # limit the number of reps per video, because some videos have too many repetitions
    dataset = RepcountDataset(data_root, split='train')
    os.makedirs(dst_dir, exist_ok=True)

    for action in CLASSES:

        for split in ['train', 'val', 'test']:
            video_list = dataset.get_video_list(split, action, max_reps=MAX_REPS)
            # build for single action
            with open(os.path.join(dst_dir, f'{action}-{split}.txt'), 'w') as f:
                for v in video_list:
                    f.write(
                        f'{v["video_path"]} {v["start"]} {v["length"]} {v["label"]}\n')

    # build for all actions
    for split in ['train', 'val', 'test']:
        video_list = dataset.get_video_list(split, action=None, max_reps=MAX_REPS)
        with open(os.path.join(dst_dir, f'all-{split}.txt'), 'w') as f:
            for v in video_list:
                action = v['class']
                if action == 'bench_pressing':
                    continue
                action_idx = CLASSES.index(action)
                label = v['label'] + 2 * action_idx
                f.write(f'{v["video_path"]} {v["start"]} {v["length"]} {label}\n')


def relabeled_csv_to_rawframe_list(csv_path: str, dst_dir: str, video_dir: str) -> None:
    """Converts relabeled csv to rawframe list.
    Creates 'train.txt', 'val.txt', and 'test.txt' files in `dst_dir`.
    Read video FPS from `video_dir`. Video path is `video_dir/{split}/{video_name}`.
    Output is in the format: `{split}/video_no_ext start_frame num_frames label`.
    
    Format of input csv::
        name,sec,label,split
        stu10_35.mp4,6.2,0,train

    Args:
        csv_path (str): Path to relabeled csv.
        dst_dir (str): Path to dir where txt files will be created.
        video_dir (str): Path to dir where video files are stored. 
        Expects 'train', 'val' and 'test' subdirs.
    """

    assert osp.isdir(video_dir), f'{video_dir} is not a directory'
    assert osp.isdir(dst_dir), f'{dst_dir} is not a directory'

    train_txt = open(osj(dst_dir, 'train.txt'), 'w')
    val_txt = open(osj(dst_dir, 'val.txt'), 'w')
    test_txt = open(osj(dst_dir, 'test.txt'), 'w')

    with open(csv_path) as f:
        lines = f.readlines()[1:]  # skip header
        for s, m, e in zip(lines[::3], lines[1::3], lines[2::3]):
            name = s.split(',')[0]
            split = s.strip().split(',')[3]
            video_path = osj(video_dir, split, name)
            if not osp.exists(video_path):
                print(f'{video_path} does not exist')
                continue
            vid = VideoReader(video_path)
            meta = vid.get_metadata()
            fps = meta['video']['fps'][0]
            start = int(float(s.split(',')[1]) * fps)
            mid = int(float(m.split(',')[1]) * fps)
            end = int(float(e.split(',')[1]) * fps)
            assert start < mid < end, f'{name}, {start} {mid} {end} not in order'
            name = name.split('.')[0]  # remove extension for rawframes folder
            num_frames_1 = mid - start + 1
            num_frames_2 = end - mid
            if split == 'train':
                train_txt.write(f'train/{name} {start} {num_frames_1} 0\n')
                train_txt.write(f'train/{name} {mid+1} {num_frames_2} 1\n')
            elif split == 'val':
                val_txt.write(f'val/{name} {start} {num_frames_1} 0\n')
                val_txt.write(f'val/{name} {mid+1} {num_frames_2} 1\n')
            elif split == 'test':
                test_txt.write(f'test/{name} {start} {num_frames_1} 0\n')
                test_txt.write(f'test/{name} {mid+1} {num_frames_2} 1\n')
    train_txt.close()
    val_txt.close()
    test_txt.close()


if __name__ == '__main__':
    print('project root:', PROJ_ROOT)
    # data_root = os.path.join(PROJ_ROOT, 'data')
    # dst_dir = os.path.join(data_root, 'Binary')
    # build_with_start(data_root, dst_dir)
    relabeled_csv_to_rawframe_list(
        '/home/umi/data/pull-up-relabeled/pull-up-relabeled.csv',
        osj(PROJ_ROOT, 'data/relabeled', 'pull_up'), osj(PROJ_ROOT,
                                                         'data/RepCount/videos'))
