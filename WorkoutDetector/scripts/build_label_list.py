import pandas as pd
import yaml
import os
from WorkoutDetector.datasets import RepcountDataset

config = yaml.safe_load(
    open(os.path.join(os.path.dirname(__file__), '../utils/config.yml')))
BASE = config['proj_root']


def build_label() -> None:
    train = os.path.join(BASE, 'data/Countix/rawframes/train.txt')
    val = os.path.join('data/Countix/rawframes/val.txt')

    traindir = os.path.join(BASE, 'data/Workouts/rawframes/Countix/train')
    valdir = os.path.join(BASE, 'data/Workouts/rawframes/Countix/val')

    traindf = pd.read_csv(os.path.join(BASE, 'datasets/Countix/workouts_train.csv'))
    valdf = pd.read_csv(os.path.join(BASE, 'datasets/Countix/workouts_val.csv'))
    classes = []
    with open(os.path.join(BASE, 'datasets/Countix/classes.txt')) as f:
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


if __name__ == '__main__':
    data_root = os.path.join(BASE, 'data')
    dst_dir = os.path.join(data_root, 'Binary')
    build_with_start(data_root, dst_dir)