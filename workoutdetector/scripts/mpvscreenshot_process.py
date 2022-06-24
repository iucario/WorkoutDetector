import os
import csv
import shutil
import pandas as pd
from typing import Tuple, List
from os.path import join as osj
import os.path as osp
from workoutdetector.settings import PROJ_ROOT, REPCOUNT_ANNO_PATH


def process_screenshot(s: str) -> Tuple[str, float]:
    """Convert mpv screenshot filename to second"""

    assert s.endswith('.png')
    name, ts = s.split('.mp4')
    name = name.split('/')[-1] + '.mp4'
    ts = ts[1:-4]
    h, m, sec = list(map(float, [ts[0:2], ts[3:5], ts[6:]]))
    return name, h * 3600 + m * 60 + sec


def test_process_screenshot():
    ax = 'stu2_48.mp4_00_00_09.943.png'
    ay = ('stu2_48.mp4', 9.943)  # name, sec
    assert process_screenshot(ax) == ay


def name_to_png(vid: str, sec: float) -> str:
    """Convert video name ans sec to mpv screenshot name"""

    h = int(sec // 3600)
    m = int(sec // 60)
    s = int(sec) % 60
    ms = str(sec).split('.')[-1].ljust(3, '0')
    return f"{vid}_{h:02}_{m:02}_{s:02}.{ms}.png"


def folder_to_csv(path: str, csv_path: str, num_frame: int = 3) -> None:
    """Convert a folder of mpv screenshots to csv. Can be used for video clipping.
    Labeled 0 and 1.

    Args:
        path: str, folder of mpv screenshots. Must contain train, val, test folders.
        csv_path: str, path to save csv
        num_frame: int, number of frames of an rep. 3 for start, mid, end.
    """

    assert os.path.isdir(path), f'{path} must be dir'
    assert osp.isdir(osj(path, 'train')), f'{path}/train must exists'
    assert osp.isdir(osj(path, 'val')), f'{path}/val must exists'
    assert osp.isdir(osj(path, 'test')), f'{path}/test must exists'
    assert num_frame == 3, f'num_frame must be 3'
    for split in ['train', 'val', 'test']:
        l = os.listdir(osj(path, split))
        assert len(
            l) % num_frame == 0, f'{split} must have divisible frames by {num_frame}'
        l.sort()
        with open(csv_path, 'w') as f:
            f.write('name,sec,label,split\n')
            for s, m, e in zip(l[::3], l[1::3], l[2::3]):
                name_0, sec_0 = process_screenshot(os.path.join(path, s))
                name_1, sec_1 = process_screenshot(os.path.join(path, m))
                name_2, sec_2 = process_screenshot(os.path.join(path, e))
                f.write(f'{name_0},{sec_0},0,{split}\n')
                f.write(f'{name_1},{sec_1},1,{split}\n')
                f.write(f'{name_2},{sec_2},0,{split}\n')


def build_general_image_dataset(path: str, out_dir: str, anno: str) -> None:
    """Creates a dir os train, val, test. Each dir contains images.
    And train.txt, val.txt, test.txt.
    
    Args:
        path: str, folder of mpv screenshots. Expects num_frame == 3 for one rep.
        start and mid will be used for class 0 and 1.
        out_dir: str, path to save csv
        anno: str, path to dataset annotation csv file.
        Information used for determining train, val, test splits.
    """

    assert os.path.isdir(path), f'{path} must be dir'
    os.makedirs(out_dir)
    os.makedirs(os.path.join(out_dir, 'train'))
    os.makedirs(os.path.join(out_dir, 'val'))
    os.makedirs(os.path.join(out_dir, 'test'))
    l = os.listdir(path)
    l.sort()

    df = pd.read_csv(anno)

    train_txt = open(os.path.join(out_dir, 'train.txt'), 'w')
    val_txt = open(os.path.join(out_dir, 'val.txt'), 'w')
    test_txt = open(os.path.join(out_dir, 'test.txt'), 'w')

    for s, m, e in zip(l[::3], l[1::3], l[2::3]):
        split = df.loc[df['name'] == s.split('.')].iloc[0]['split']
        shutil.copy(os.path.join(path, s), os.path.join(out_dir, split, s))
        shutil.copy(os.path.join(path, m), os.path.join(out_dir, split, m))
        shutil.copy(os.path.join(path, e), os.path.join(out_dir, split, e))
        if split == 'train':
            train_txt.write(f'{s}\n')
            train_txt.write(f'{m}\n')
        elif split == 'val':
            val_txt.write(f'{s}\n')
            val_txt.write(f'{m}\n')
        else:
            test_txt.write(f'{s}\n')
            test_txt.write(f'{m}\n')


def build_image_folder(path: str, out_dir: str) -> None:
    """Build folder for ImageFolder dataset"""

    assert len(os.listdir(path)) % 2 == 0, f'{path} files not even'
    os.makedirs(out_dir)
    os.makedirs(os.path.join(out_dir, '0'))
    os.makedirs(os.path.join(out_dir, '1'))
    l = os.listdir(path)
    l.sort()
    for s, e in zip(l[::2], l[1::2]):
        n1, s1 = process_screenshot(s)
        n2, s2 = process_screenshot(e)
        assert n1 == n2, f'{n1} != {n2} Name not equal'
        assert s1 < s2, f'{s1} !< {s1} Time not in order'
        shutil.copy(os.path.join(path, s), os.path.join(out_dir, '0', s))
        shutil.copy(os.path.join(path, e), os.path.join(out_dir, '1', e))


def build_label(path: str, out_dir: str, anno_path: str) -> None:
    """Build label file for ImageFolder
    Creates train.txt, val.txt, test.txt in out_dir.

    Args:
        path: str, ImageFolder. Has subfolders with class names
        out_dir: str, dir to save label files
        anno_path: str, path to RepCount dataset annotation.csv
    """

    df = pd.read_csv(anno_path)
    train_txt = open(os.path.join(out_dir, 'train.txt'), 'w')
    val_txt = open(os.path.join(out_dir, 'val.txt'), 'w')
    test_txt = open(os.path.join(out_dir, 'test.txt'), 'w')
    os.makedirs(os.path.join(out_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'test'), exist_ok=True)

    folders = [x for x in os.listdir(path) if os.path.isdir(osj(path, x))]
    for folder in folders:
        for img in os.listdir(os.path.join(path, folder)):
            name, sec = process_screenshot(img)
            split = df[df['name'] == name]['split'].values
            if split:
                split = split[0]
            else:
                print(name)
                continue
            if split == 'train':
                train_txt.write(f'{name} {folder}\n')
            elif split == 'val':
                val_txt.write(f'{name} {folder}\n')
            else:
                test_txt.write(f'{name} {folder}\n')
    train_txt.close()
    val_txt.close()
    test_txt.close()


def label_from_folder(path: str, out_dir: str, anno: str) -> None:
    """Given train val folders, create label txt files"""

    assert osp.isdir(osj(path, 'train')), f'{osj(path, "train")} not dir'
    assert osp.isdir(osj(path, 'val')), f'{osj(path, "val")} not dir'
    df = pd.read_csv(anno, names=['name', 'sec', 'class_'], sep=' ')
    with open(osj(path, 'train.txt'), 'w') as f:
        for x in os.listdir(osj(path, 'train')):
            name, sec = process_screenshot(x)
            class_ = df[(df['name'] == name) & (df['sec'] == sec)]['class_'].values[0]
            png_name = name_to_png(name, sec)
            assert osp.exists(osj(path, str(class_),
                                  png_name)), f"{class_}/{png_name} does not exist"
            f.write(f"{png_name} {class_}\n")
    with open(osj(path, 'val.txt'), 'w') as f:
        for x in os.listdir(osj(path, 'val')):
            name, sec = process_screenshot(x)
            class_ = df[(df['name'] == name) & (df['sec'] == sec)]['class_'].values[0]
            png_name = name_to_png(name, sec)
            assert osp.exists(osj(path, str(class_),
                                  png_name)), f"{class_}/{png_name} does not exist"
            f.write(f"{png_name} {class_}\n")


def label_from_split(path: str) -> None:
    """From path/train, path/val, path/test, creates label files
    saved in path/train.txt, path/val.txt, path/test.txt

    Args:
        path: str, path to dir 
    """

    assert osp.isdir(osj(path, 'train')), f'{osj(path, "train")} not dir'
    assert osp.isdir(osj(path, 'val')), f'{osj(path, "val")} not dir'
    assert osp.isdir(osj(path, 'test')), f'{osj(path, "test")} not dir'
    for split in ['train', 'val', 'test']:
        with open(osj(path, split + '.txt'), 'w') as f:
            l = os.listdir(osj(path, split))
            l.sort()
            for s, m, e in zip(l[::3], l[1::3], l[2::3]):
                f.write(f'{split}/{s} 0\n')
                f.write(f'{split}/{m} 1\n')


def main():
    """How to prepare data

    1. The mpv screenshot filename template is `screenshot-template=~/Desktop/%f_%P`
    2. Will get files like `stu2_48.mp4_00_00_09.943.png`
    3. If saved in train, val, test folders, use label_from_split(root_dir)
    4. If screenshots are saved in one folder, I need to write a new script.
    5. And `folder_to_csv` can save timestamps to csv for future usage.
    """

    label_from_split('/home/umi/data/pull-up-relabeled/')


if __name__ == '__main__':
    # folder_to_csv('/home/umi/data/pull-up-relabeled',
    #               '/home/umi/data/pull-up-relabeled/pull-up-relabeled.csv')
    # build_image_folder('/home/umi/tmp/situp', '/home/umi/data/situp')
    # build_label('/home/umi/data/pull-up-relabeled/', '/home/umi/data/pull-up-relabeled/',
    #             REPCOUNT_ANNO_PATH)
    main()
