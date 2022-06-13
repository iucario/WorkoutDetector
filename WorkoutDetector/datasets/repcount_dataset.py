import math
import os
from typing import List, Tuple
import einops
import torch
from torch import Tensor
import cv2
import pandas as pd
import numpy as np
import base64
from torchvision.datasets.utils import download_url, download_and_extract_archive, verify_str_arg
from torchvision.io import read_image


def parse_onedrive(link) -> str:
    """Parse onedrive link to download link.

    Args:
        link: str, start with `https://1drv.ms/u/s!`

    Returns:
        str, download link.
    """
    b = base64.urlsafe_b64encode(link.strip().encode('ascii'))
    s = b.decode('ascii')
    res = f'https://api.onedrive.com/v1.0/shares/u!{s}/root/content'
    return res


def sample_frames(total: int, num: int, offset=0):
    """Uniformly sample num frames from video
    
    Args:
        total: int, total frames, 
        num: int, number of frames to sample
        offset: int, offset from start of video
    Returns: 
        list of frame indices starting from offset
    """

    if total < num:
        # repeat frames if total < num
        repeats = math.ceil(num / total)
        data = [x for x in range(total) for _ in range(repeats)]
        total = len(data)
    else:
        data = list(range(total))
    interval = total // num
    indices = np.arange(0, total, interval)[:num]
    for i, x in enumerate(indices):
        rand = np.random.randint(0, interval)
        if i == num - 1:
            upper = total
            rand = np.random.randint(0, upper - x)
        else:
            upper = min(interval * (i + 1), total)
        indices[i] = (x + rand) % upper
    assert len(indices) == num, f'len(indices)={len(indices)}'
    for i in range(1, len(indices)):
        assert indices[i] > indices[i - 1], f'indices[{i}]={indices[i]}'
    return [data[i] + offset for i in indices]


class RepcountDataset(torch.utils.data.Dataset):
    """Repcount dataset
    https://github.com/SvipRepetitionCounting/TransRAC
    
    Args:
        root: str, root dir
        split: str, train or val or test
    
    Notes:
        File tree::

            |- RepCount
            |   |- annotation.csv
            |   |- rawframes
            |   |   |- train
            |   |   |     |- video_name/img_00001.jpg
            |   |   |- val
            |   |   |- test
            |   |- videos
            |       |- train
            |       ...

        The annotation csv file has columns:
            name: video name, e.g. 'video_1.mp4'
            class: action class name, e.g. 'squat'
            vid: YouTube video id of length 11
            start: start frame index
            end: end frame index
            count: repetition count
            reps: repetition indices, in format `[s1, e1, s2, e2, ...]`

    """
    _URL_VIDEO = 'https://1drv.ms/u/s!AiohV3HRf-34ipk0i1y2P1txpKYXFw'
    _URL_ANNO = 'https://1drv.ms/f/s!AiohV3HRf-34i_V9MWtdu66tCT2pGQ'
    _URL_RAWFRAME = 'https://1drv.ms/u/s!AiohV3HRf-34ipwACYfKSHhkZzebrQ'

    def __init__(self, root, split='train', transform=None, download=False) -> None:
        super(RepcountDataset, self).__init__()
        self._data_path = os.path.join(root, 'RepCount')
        if download:
            self._download()
        if not os.path.isdir(self._data_path):
            raise RuntimeError(
                "Dataset not found or corrupted. You can use download=True to download it"
            )
        verify_str_arg(split, "split", ("train", "val", "test"))
        self.split = split
        anno_path = os.path.join(self._data_path, 'annotation.csv')
        if not os.path.exists(anno_path):
            raise OSError(
                f'{anno_path} not found. Consider move '\
                    '`/datasets/RepCount/all_data.csv` to `{anno_path}`')
        anno_df = pd.read_csv(anno_path, index_col=0)
        self.df = anno_df[anno_df['split'] == split]
        self.classes = self.df['class_'].unique().tolist()
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[str, int]:
        """Returns path to video rawframe and video class.
        For action recognition.
        """

        row = self.df.iloc[index]
        video_frame_path = os.path.join(self._data_path, 'rawframes', row['split'],
                                        row['name'])
        label = self.classes.index(row['class'])
        count = row['count']
        reps = list(map(int, row.reps.split())) if count else []
        return video_frame_path, label

    def __len__(self) -> int:
        return len(self.df)

    def _download(self) -> None:
        """
        Download the RepCount dataset archive from OneDrive and extract it under root.
        """
        if self._check_exists():
            return
        # the extracted folder is `rawframes`, may upload again sometime
        download_and_extract_archive(self._URL_RAWFRAME,
                                     download_root=self._data_path,
                                     filename='rawframes.zip',
                                     extract_root=self._data_path)

    def _check_exists(self) -> bool:
        return os.path.exists(self._data_path) and os.path.isdir(self._data_path)


class RepcountImageDataset(RepcountDataset):
    """Repcount image dataset for binary classification of start and end state in specific action

    Args:
        action: str
    """

    def __init__(self,
                 root: str,
                 action: str,
                 split='train',
                 transform=None,
                 download=False) -> None:
        super(RepcountImageDataset, self).__init__(root, split, transform, download)
        verify_str_arg(action, "action", self.classes)
        self.df = self.df[self.df['class_'] == action]
        images = []
        labels = []
        for row in self.df.itertuples():
            if row['count'] == 0:
                continue
            name = row.name.split('.')[0]
            reps = list(map(int, row.reps.split()))
            for start, end in zip(reps[::2], reps[1::2]):
                start, end = start + 1, end + 1
                mid = (start + end) // 2
                images.append(f'{name}/img_{start:05}.jpg')
                images.append(f'{name}/img_{mid:05}.jpg')
                labels.append(0)
                labels.append(1)
        self.images = images
        self.labels = labels
        self.action = action
        self._prefix = os.path.join(self._data_path, 'rawframes', split)

    def __getitem__(self, index: int) -> tuple:
        img_path = os.path.join(self._prefix, self.images[index])
        img = read_image(img_path)
        label = self.labels[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self) -> int:
        return len(self.images)


class RepcountVideoDataset(RepcountDataset):
    """Binary classification of start and end state in specific action. Using video as input.
    
    It's like `RepcountImageDataset`, but using multiple frames rather than only two images.

    Args:
        action: str
        num_frames: int, number of frames in one video
    """

    def __init__(self,
                 root: str,
                 action: str,
                 num_segments: int = 8,
                 split='train',
                 transform=None,
                 download=False) -> None:
        super(RepcountVideoDataset, self).__init__(root, split, transform, download)
        verify_str_arg(action, "action", self.classes)
        self.df = self.df[self.df['class_'] == action]
        self.video_list = self.setup()
        self.num_segments = num_segments

    def setup(self) -> List[dict]:
        """
        Returns:
            list of dict: videos, 
                {
                    video_path: path_to_raw_frames_dir, 
                    start: start_frame_index, start from 1,
                    end: end_frame_index
                    length: end_frame_index - start_frame_index + 1
                    class: action class,
                    label: 0 or 1
                }
        """

        videos = []
        for row in self.df.itertuples():
            name = row.name.split('.')[0]
            count = row.count
            reps = list(map(int, row.reps.split())) if count > 0 else []
            for start, end in zip(reps[0::2], reps[1::2]):
                start += 1  # plus 1 because img index starts from 1
                end += 1  # but annotated frame index starts from 0
                mid = (start + end) // 2
                videos.append({
                    'video_path':
                        os.path.join(self._data_path, 'rawframes', self.split, name),
                    'start':
                        start,
                    'end':
                        mid,
                    'length':
                        mid - start + 1,
                    'class':
                        row.class_,
                    'label':
                        0
                })
                videos.append({
                    'video_path':
                        os.path.join(self._data_path, 'rawframes', self.split, name),
                    'start':
                        mid + 1,
                    'end':
                        end,
                    'length':
                        end - mid,
                    'class':
                        row.class_,
                    'label':
                        1
                })
        return videos

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        frame_list = []
        start = self.video_list[index]['start']
        length = self.video_list[index]['length']
        samples = sample_frames(length, self.num_segments, start)
        for i in samples:
            frame_path = os.path.join(self.video_list[index]['video_path'],
                                      f'img_{i:05}.jpg')
            frame = read_image(frame_path)
            frame_list.append(frame)
        if self.transform is not None:
            frame_list = [self.transform(frame) for frame in frame_list]
        frame_list = torch.stack(frame_list, 0)
        assert frame_list.shape[0] == self.num_segments, \
            f'frame_list.shape[0] = {frame_list.shape[0]}, ' \
            f'but self.num_segments = {self.num_segments}'
        return frame_list, self.video_list[index]['label']

    def __len__(self) -> int:
        return len(self.video_list)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    data_root = '/home/umi/projects/WorkoutDetector/data'
    dataset = RepcountVideoDataset(data_root, split='test', action='push_up')
    print(dataset.classes)
    # imageset = RepcountImageDataset(data_root, action='jump_jack', split='test')
    random_index = np.random.randint(0, len(dataset))
    img, label = dataset[random_index]
    plt.figure(figsize=(8, 4),dpi=200)
    img = einops.rearrange(img, '(b1 b2) c h w -> (b1 h) (b2 w) c', b1=2)
    plt.title(f'label: {label}')
    print(img.shape)
    plt.imshow(img)
    plt.show()