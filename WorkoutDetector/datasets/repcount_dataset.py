import os
import torch
import cv2
import pandas as pd
import numpy as np
import base64
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
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


class RepcountDataset(torch.utils.data.Dataset):
    """Repcount raw frames dataset
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

        The annotation csv file has columns:
            name: video name, e.g. 'video_1.mp4'
            class: action class name, e.g. 'squat'
            vid: YouTube video id of length 11
            start: start frame index
            end: end frame index
            count: repetition count
            reps: repetition indices, in format `[s1, e1, s2, e2, ...]`

    """
    _URL = ''

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
        anno_path = os.path.join(self._data_path, 'annotation.csv')
        anno_df = pd.read_csv(anno_path, index_col=0)
        self.df = anno_df[anno_df['split'] == split]
        self.classes = self.df['class'].unique()
        self.transform = transform

    def __getitem__(self, index: int) -> tuple:

        pass

    def __len__(self) -> int:
        return len(self.df)

    def _download(self) -> None:
        """
        Download the RepCount dataset archive from OneDrive and extract it under root.
        """
        if self._check_exists():
            return
        download_and_extract_archive(self._URL, download_root=self._data_path)

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
        self.df = self.df[self.df['class'] == action]
        images = []
        labels = []
        for row in self.df.itertuples():
            if row.count == 0:
                continue
            name = row.name.split('.')[0]
            reps = list(map(int, row.reps.split()))
            for start, end in zip(reps[::2], reps[1::2]):
                start, end = start+1, end+1
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


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    data_root = '/home/umi/projects/WorkoutDetector/data'
    # dataset = RepcountDataset(data_root, split='test')
    imageset = RepcountImageDataset(data_root, action='bench_pressing', split='test')
    print(len(imageset))
    print(imageset.classes)
    # random_index = np.random.randint(0, len(imageset))
    # img, label = imageset[random_index]
    # plt.imshow(img.permute(1, 2, 0))
    # print(label)
    # plt.show()