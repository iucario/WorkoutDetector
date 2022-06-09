import os
import torch
import cv2
import pandas as pd
import numpy as np


class RepcountDataset(torch.utils.data.Dataset):
    """Repcount raw frames dataset
    https://github.com/SvipRepetitionCounting/TransRAC
    
    Args:
        root: str, root dir
    """

    def __init__(self, root, split='train', transform=None, download=False):
        super(RepcountDataset, self).__init__()
        self._data_path = os.path.join(root, 'RepCount')
        anno_path = os.path.join(self._data_path, 'annotation.csv')
        anno_df = pd.read_csv(anno_path, index_col=0)

    def _download(self) -> None:
        """
        Download the RepCount dataset archive from OneDrive and extract it under root.
        """
        if self._check_exists():
            return
        # TODO

    def _check_exists(self) -> bool:
        return os.path.exists(self._data_path) and os.path.isdir(self._data_path)


if __name__ == '__main__':
    dataset = RepcountDataset('/home/umi/projects/WorkoutDetector/data')