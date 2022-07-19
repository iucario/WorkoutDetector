import os
from typing import Union
import pandas as pd
import cv2
from torchvision.io import read_video, write_video, VideoReader
from os.path import join as osj


def cut_video(video: str,
              start: Union[float, int],
              end: Union[float, int],
              out_file: str,
              mode: str = 'frame') -> None:
    """Cut video from start to end and save to file.

    Args:
        video (str): path to video
        start (int or float): start time in frame index or seconds
        end (int or float): end time in frame index or seconds
        out_file (str): path to output video
        mode (str): 'frame' or 'sec'.
    """

    vr = VideoReader(video)
    meta = vr.get_metadata()
    fps = meta['video']['fps'][0]
    if mode == 'frame':
        start_sec, end_sec = start / fps, end / fps
    elif mode == 'sec':
        start_sec, end_sec = start, end
    vid, _, _ = read_video(video, start_sec, end_sec, 'sec')
    print(f'Writing {out_file}, fps: {fps}, start: {start}, end: {end}, {vid.shape}')
    write_video(out_file, vid, fps)


def build_video(data_root: str, label_file: str, dest_dir: str) -> None:
    """Cut untrimmed videos and save to dest_dir.

    Args:
        data_root (str): path to data root
        label_file (str): path to label file. Contains 
            (video_name, start_frame, length, label)
        dest_dir (str): path to destination directory. Videos will be renamed to
            `{video_name}_{start_frame}.mp4`
    """
    for split in ['train', 'test', 'val']:
        os.makedirs(osj(dest_dir, split), exist_ok=True)
    with open(label_file) as f:
        for line in f:
            vid, start, length, label = line.strip().split(' ')
            name = os.path.basename(vid)
            if not vid.endswith('.mp4'):
                vid += '.mp4'
            video_path = osj(data_root, vid)
            out_path = osj(dest_dir, f'{name}_{start}.mp4')
            cut_video(video_path, int(start), int(start) + int(length), out_path)
    print(f'Done! Videos are saved to {dest_dir}')


def build_video_rep(data_root: str, anno_path: str, dest_dir: str) -> None:
    """Cut videos to rep states. Matches the SlowFast Kinetics dataset format.
    Specifically, RepCount dataset 12 classes.
    Generates label files `train.csv`, `val.csv`, `test.csv` in `dest_dir`.
    Use OpenCV to read frame by frame and break at rep end.
    
    Args:
        data_dir: path like `data/RepCount/videos`. Expects train,val,test subfolders in it.
        anno_path: csv file path
        dest_dir: cutted videos will be saved in `dest_dir/{split}/{name}.mp4`. 

    Example:
        >>> data_dir = '~/data/RepCount/video'
        >>> anno_path = '~/data/RepCount/annotation.csv'
        >>> dest_dir = '~/data/RepCount/rep_video'
        >>> build_video_rep(data_dir, anno_path, dest_dir)
        # first line in train.txt: train/train951_0.mp4 10
    """

    os.makedirs(dest_dir, exist_ok=True)
    for split in ['train', 'val', 'test']:
        if not os.path.exists(os.path.join(dest_dir, split)):
            os.makedirs(os.path.join(dest_dir, split))
        if os.path.isfile(os.path.join(dest_dir, f'{split}.csv')):
            # remove existing csv file because we are appending to it
            os.remove(os.path.join(dest_dir, f'{split}.csv'))

    train_csv = open(os.path.join(dest_dir, 'train.csv'), 'a')
    val_csv = open(os.path.join(dest_dir, 'val.csv'), 'a')
    test_csv = open(os.path.join(dest_dir, 'test.csv'), 'a')

    CLASSES = ['situp', 'push_up', 'pull_up', 'jump_jack', 'squat', 'front_raise']
    anno = pd.read_csv(anno_path)
    for i, row in anno.iterrows():

        if row['class_'] not in CLASSES:
            continue

        count = int(row['count'])
        if count == 0:
            continue

        split = row["split"]
        video_path = os.path.join(data_root, split, row['name'])
        video_name = row['name']

        reps = [int(x) for x in row['reps'].split()]
        start_frame = reps[0]
        end_frame = reps[1]  # Select one sample from the one video
        mid_frame = (start_frame + end_frame) // 2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        name_no_ext = video_name.split('.')[0]
        out_1 = cv2.VideoWriter(os.path.join(dest_dir, split, f'{name_no_ext}_0.mp4'),
                                cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        out_2 = cv2.VideoWriter(os.path.join(dest_dir, split, f'{name_no_ext}_1.mp4'),
                                cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        frame_idx = 0
        while (cap.isOpened()):
            ret, frame = cap.read()
            if not ret or frame_idx == end_frame:
                break
            if start_frame <= frame_idx < mid_frame:
                out_1.write(frame)
            elif mid_frame <= frame_idx < end_frame:
                out_2.write(frame)

            frame_idx += 1
        cap.release()
        out_1.release()
        out_2.release()

        rep_class = CLASSES.index(row["class_"]) * 2
        if split == 'train':
            train_csv.write(f'{split}/{name_no_ext}_0.mp4 {rep_class}\n')
            train_csv.write(f'{split}/{name_no_ext}_1.mp4 {rep_class + 1}\n')
        elif split == 'val':
            val_csv.write(f'{split}/{name_no_ext}_0.mp4 {rep_class}\n')
            val_csv.write(f'{split}/{name_no_ext}_1.mp4 {rep_class + 1}\n')
        elif split == 'test':
            test_csv.write(f'{split}/{name_no_ext}_0.mp4 {rep_class}\n')
            test_csv.write(f'{split}/{name_no_ext}_1.mp4 {rep_class + 1}\n')
        print(f'{video_path} done')

    print('Done')


if __name__ == '__main__':
    data_root = os.path.expanduser('~/data/RepCount/videos')
    anno_path = 'datasets/RepCount/annotation.csv'
    dest_dir = os.path.expanduser('~/data/RepCount/action_video/train')
    label_file = os.path.expanduser('~/data/RepCount/rawframes/train-action.txt')
    # build_video_rep(data_root, anno_path, dest_dir)
    build_video(data_root, label_file, dest_dir)