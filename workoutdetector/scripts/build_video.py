import os
import pandas as pd
import cv2


def build_video_rep(data_dir: str, anno_path: str, dest_dir: str) -> None:
    """Cut videos to rep states. Matches the SlowFast Kinetics dataset format.
    Specifically, RepCount dataset 12 classes.
    Generates label files `train.csv`, `val.csv`, `test.csv` in `dest_dir`.
    Use OpenCV to read frame by frame and break at rep end.
    
    Args:
        data_dir: path like `data/RepCount/videos`. Expects train,val,test subfolders in it.
        anno_path: csv file path
        dest_dir: cutted videos will be saved int `dest_dir/split/video_name`. 

    Example:
        >>> data_dir = '~/data/RepCount/video'
        >>> anno_path = '~/data/RepCount/annotation.csv'
        >>> dest_dir = '~/data/RepCount/rep_video'
        >>> build_video_rep(data_dir, anno_path, dest_dir)
        # first line in train.txt: train/train951_0.mp4 10
    """

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
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
        video_path = os.path.join(data_dir, split, row['name'])
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
    data_dir = 'data/RepCount/videos'
    anno_path = 'datasets/RepCount/annotation.csv'
    dest_dir = 'data/RepCount/rep_video'
    build_video_rep(data_dir, anno_path, dest_dir)