import json
import os
import os.path as osp
from os.path import join as osj

import pandas as pd
from torchvision.io import VideoReader
from workoutdetector.datasets import RepcountDataset, RepcountHelper
from workoutdetector.settings import PROJ_ROOT


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


def build_with_start(data_root: str, anno_file: str, dst_dir: str) -> None:
    """Creates label files for repetition video classification.
    Example line: `/app/data/RepCount/rawframes/test/train3344 2 12 0`

    Args:
        data_root (str): Directory to RepCount dataset. Same as in RepCountDataset.
        anno_file (str): Path to annotation file.
        dst_dir (str): Path to dir where txt files will be created.

    Note:
        For each action and `all-{split}.txt` file is created.
        Lines in the label files are in the format:
            `{video_path} {start} {length} {label}`
    
    Example::

        >>> anno_file = 'datasets/RepCount/annotation.csv'
        >>> data_root = os.path.expandeuser('data')
        >>> dst_dir = os.path.join(data_root, 'Binary')
        >>> build_with_start(data_root, anno_file, dst_dir)

        project root: /work
        ===> Done! Label files are created in
        /home/user/data/Binary
        ===> First line of all-train.txt:
        RepCount/rawframes/train/train951 7 34 10
        # Creates `{data_root}/Binary/all-train.txt` and `{data_root}/Binary/all-val.txt` and 
        # `all-test.txt`, `situp-train.txt`, etc.
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

    print(f'===> Done! Label files are created in\n{dst_dir}')
    first_line = open(osp.join(dst_dir, 'all-train.txt')).readline()
    print(f'===> First line of all-train.txt:\n{first_line}')


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

    Example:
        >>> relabeled_csv_to_rawframe_list(
            '/home/umi/data/pull-up-relabeled/pull-up-relabeled.csv',
            osj(PROJ_ROOT, 'data/relabeled', 'pull_up'), 
            osj(PROJ_ROOT, 'data/RepCount/videos'))
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


def activity_net_label(csv: str, outdir: str) -> None:
    """Creates a JSON file with activity net labels.
    Used in MMAction2.

    Annotation format::

        {
            "v_--1DO2V4K74":  {
                "duration_second": 211.53,
                "duration_frame": 6337,
                "annotations": [
                    {
                        "segment": [
                            30.025882995319815,
                            205.2318595943838
                        ],
                        "label": "Rock climbing"
                    }
                ],
                "feature_frame": 6336,
                "fps": 30.0,
                "rfps": 29.9579255898
            },
            "v_--6bJUbfpnQ": {
                "duration_second": 26.75,
                "duration_frame": 647,
                "annotations": [
                    {
                        "segment": [
                            2.578755070202808,
                            24.914101404056165
                        ],
                        "label": "Drinking beer"
                    }
                ],
                "feature_frame": 624,
                "fps": 24.0,
                "rfps": 24.1869158879
            },
            ...
        }

    Args:
        csv (str): Path to csv file.
        outdir (str): Dir to save JSON file.
    Example:
        >>> root = osj(PROJ_ROOT, 'data/RepCount/videos')
        >>> csv = osj(root, 'annotation.csv')
        >>> outdir = osj(root, 'activity')
        >>> activity_net_label(csv, outdir)
        train done. 545 videos
        val done. 100 videos
        test done. 117 videos
        Done
    """
    if not osp.exists(outdir):
        os.makedirs(outdir)

    helper = RepcountHelper(anno_file=csv, data_root='/home/umi/data/RepCount')
    for split in ['train', 'val', 'test']:
        with open(osj(outdir, f'{split}.json'), 'w') as f:
            vids = helper.get_rep_data(action=['all'], split=[split])
            data = {}
            for vid in vids.values():
                vr = VideoReader(osp.join(helper.data_root, vid.video_path))
                meta = vr.get_metadata()
                fps = meta['video']['fps'][0]
                anno = []
                for start, end in zip(vid.reps[::2], vid.reps[1::2]):
                    anno.append({
                        'segment': [start * fps, end * fps],
                        'label': vid.class_
                    })
                data[vid.video_name] = {
                    'duration_second': meta['video']['duration'][0],
                    'rfps': fps,
                    'fps': round(fps),
                    'duration_frame': vid.total_frames,
                    'feature_frame': vid.total_frames - 1,  # I don't know what this is
                    'annotations': anno
                }
            json.dump(data, f, indent=4)
            print(f'{split} done. {len(data)} videos')
    print('Done')


if __name__ == '__main__':
    print('project root:', PROJ_ROOT)
    anno_file = 'datasets/RepCount/annotation.csv'
    data_root = osp.expanduser('~/data')
    # dst_dir = os.path.join(data_root, 'Binary')
    # build_with_start(data_root, anno_file, dst_dir)
    dst = osj(data_root, 'RepCount/activity')
    activity_net_label(anno_file, dst)

    # relabeled_csv_to_rawframe_list(
    #     '/home/umi/data/pull-up-relabeled/pull-up-relabeled.csv',
    #     osj(PROJ_ROOT, 'data/relabeled', 'pull_up'), osj(PROJ_ROOT,
    #                                                      'data/RepCount/videos'))
