import sys
import os
import os.path as osp
import pandas as pd
import yaml

config = yaml.safe_load(open('config.yml', 'r'))


class Repcount:
    def __init__(self):
        self.anno_root = osp.join(config['proj_root'], 'datasets/RepCount/annotation')
        self.anno_train_path = osp.join(self.anno_root, 'train.csv')
        self.anno_val_path = osp.join(self.anno_root, 'val.csv')
        self.anno_test_path = osp.join(self.anno_root, 'test.csv')
        self.anno_all = pd.concat(self.get_anno(split)
                                  for split in ['train', 'val', 'test'])

    def get_anno(self, split='train'):
        split = split.lower()
        if split == 'train':
            anno = pd.read_csv(self.anno_train_path, delimiter=',')
        elif split == 'val':
            anno = pd.read_csv(self.anno_val_path, delimiter=',')
        elif split == 'test':
            anno = pd.read_csv(self.anno_test_path, delimiter=',')
        else:
            raise ValueError(f'Invalid split: {split}. Must be one of train, val, test')
        return anno

    def get_reps(self, name):
        """Input: name
        Returns: list of frames in order of [start_1, end_1, start_2, end_2, ...]"""
        if name not in self.anno_all.name.unique():
            raise ValueError(f'Video {name} not found in annotation')
        isnum = self.anno_all[self.anno_all['name']
                              == name.strip()].iloc[0].dropna()
        row = isnum.values[4:].astype(int)
        return list(row)

    def get_video(self, name):
        """Input: name
        Returns: video_path"""
        pass

    def get_pose(self, name):
        """Input: name
        Returns: pose of the video"""
        pass

    def __repr__(self):
        return f'Repcount(anno_root={self.anno_root})\n'\
            f'anno_train_path={self.anno_train_path}\n'\
            f'anno_val_path={self.anno_val_path},\n'\
            f'anno_test_path={self.anno_test_path}\n'\
            f'len_anno_all={len(self.anno_all)}\n'\
            f'columns={self.anno_all.columns[:10]}\n'


if __name__ == '__main__':
    repcount = Repcount()
    print(repcount)
    x = repcount.get_reps('stu2_39.mp4')
    print(x)
