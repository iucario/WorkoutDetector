from WorkoutDetector.utils.common import Repcount
from WorkoutDetector.datasets import RepcountVideoDataset
import os
import shutil
import yaml
import tqdm

config = yaml.safe_load(
    open(os.path.join(os.path.dirname(__file__), '../utils/config.yml')))
BASE = config['proj_root']

class_name = 'squat'

repcount = Repcount()


def build_split(classname, split='train'):
    """Copy images to binary classification directory.

    Example:
        |- data/Binary/squat
            |- train
                |- vid_1_start-frame
                |- img_00001.jpg
                |- img_00002.jpg
            |- val
            |- vid_2_start-frame

        start - mid: 0, mid - end: 1
    """

    dst_data_dir = os.path.join(BASE, f'data/Binary/{classname}')
    df = repcount.get_anno(split)
    df = df[df['type'] == classname]
    data_root = os.path.join(repcount.data_root, 'rawframes', split)
    vids = df['name'].values

    # for vid in tqdm(vids):
    #     vid = vid.split('.')[0]
    #     count, reps = repcount.get_count(vid)
    #     for start, end in zip(reps[0::2], reps[1::2]):
    #         start += 1
    #         end += 1
    #         mid = (start + end) // 2
    #         os.mkdir(os.path.join(dst_data_dir, f'{vid}_{start}'))
    #         os.mkdir(os.path.join(dst_data_dir, f'{vid}_{mid}'))
    #         for i in range(start, mid):
    #             j = i-start+1
    #             src_path = os.path.join(data_root, f'{vid}/img_{i:05}.jpg')
    #             dst_path = os.path.join(dst_data_dir, f'{vid}_{start}/img_{j:05}.jpg')
    #             shutil.copy(src_path, dst_path)
    #         for i in range(mid, end):
    #             j = i-mid+1
    #             src_path = os.path.join(data_root, f'{vid}/img_{i:05}.jpg')
    #             dst_path = os.path.join(dst_data_dir, f'{vid}_{mid}/img_{j:05}.jpg')
    #             shutil.copy(src_path, dst_path)

    label_file = os.path.join(BASE, f'data/Binary/{classname}-{split}.txt')
    with open(label_file, 'w') as f:
        for vid in vids:
            vid = vid.split('.')[0]
            count, reps = repcount.get_count(vid)
            for start, end in zip(reps[0::2], reps[1::2]):
                start += 1
                end += 1
                mid = (start + end) // 2
                f.write(f'{classname}/{vid}_{start} {mid-start} 0\n')
                f.write(f'{classname}/{vid}_{mid} {end-mid} 1\n')


def rename_images():
    data_dir = os.path.join(BASE, f'data/Binary/{class_name}')
    for d in os.listdir(data_dir):
        for img in os.listdir(os.path.join(data_dir, d)):
            offset = int(d.split('_')[2])
            img_id = int(img.split('.')[0].split('_')[1])
            new_id = img_id - offset + 1
            new_name = f'{new_id:05}.jpg'
        for img in os.listdir(os.path.join(data_dir, d)):
            new_name = f'img_{img}'
            os.rename(os.path.join(data_dir, d, img), os.path.join(data_dir, d, new_name))


def build_with_start():
    CLASSES = [
        'situp', 'push_up', 'pull_up', 'bench_pressing', 'jump_jack', 'squat',
        'front_raise'
    ]
    data_root = os.path.join(BASE, 'data')
    DEST_DIR = os.path.join(data_root, 'Binary')
    for action in CLASSES:
        train_set = RepcountVideoDataset(root=data_root,
                                         action=action,
                                         split='train',
                                         transform=None)
        val_set = RepcountVideoDataset(root=data_root,
                                       action=action,
                                       split='val',
                                       transform=None)
        test_set = RepcountVideoDataset(root=data_root,
                                        action=action,
                                        split='test',
                                        transform=None)

        for split, split_set in zip(['train', 'val', 'test'],
                                    [train_set, val_set, test_set]):
            # build for single action
            with open(os.path.join(DEST_DIR, f'{action}-{split}.txt'), 'w') as f:
                for v in split_set.video_list:
                    f.write(
                        f'{v["video_path"]} {v["start"]} {v["length"]} {v["label"]}\n')

            # build for all actions
            with open(os.path.join(DEST_DIR, f'all-{split}.txt'), 'a') as f:
                for v in split_set.video_list:
                    action_idx = CLASSES.index(action)
                    label = v['label'] + 2 * action_idx
                    f.write(f'{v["video_path"]} {v["start"]} {v["length"]} {label}\n')


if __name__ == '__main__':
    build_with_start()