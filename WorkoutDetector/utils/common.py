import random
import sys
import os
import os.path as osp
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import yaml
import sklearn.metrics.pairwise
import seaborn as sns

# sys.path.append(osp.join(osp.dirname(__file__), '..'))
from utils.visualize import Vis2DPose

config = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), 'config.yml')))

KPS = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder',
       'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
       'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
CLASSES = ['front_raise', 'pull_up', 'squat', 'bench_pressing', 'jumping_jack', 'situp',
           'push_up', 'battle_rope', 'exercising_arm', 'lunge', 'mountain_climber']


def gen_gif(video_path, anno):
    """Generate 2D Skeletons with the original RGB Video to `tmp.gif`"""
    frame_dir = anno['frame_dir']
    vid = Vis2DPose(anno, thre=0.5, out_shape=(540, 960),
                    layout='coco', fps=20, video=video_path)
    vid.to_gif('tmp.gif')


def plot_time_series(item):
    """Plot time series of a single pose item
        item: dict, {'frame_dir', 'label', 'keypoint', 'keypoint_score',
         'img_shape', 'num_person_raw', 'total_frames', 'count'}
    """
    name = item['frame_dir']
    kp = item['keypoint'][0]  # first person
    num_frame = kp.shape[0]
    plt.plot(range(num_frame), kp[:, 0, :], label='x')  # nose
    plt.legend(['x axis', 'y axis'])
    plt.xlabel('frame')
    plt.ylabel('position')
    plt.title(name+str(item['count']))
    plt.show()


def plot_pose_heatmap(item):
    """Cosine similarity between 2D poses heatmap
        item: Pose item
        reps: list of repetition starts and ends"""
    kp = item['keypoint'][0]
    num_frame = kp.shape[0]
    mat = np.zeros((num_frame, num_frame))
    feat = kp.reshape(num_frame, -1)
    mat = sklearn.metrics.pairwise.pairwise_distances(feat, metric='cosine')
    sns.heatmap(mat, cmap='viridis')
    plt.title(f"pose {item['frame_dir']} {item['count']} reps")
    if 'reps' in item:
        plt.vlines(item['reps'][::2], colors='r', ymin=0, ymax=len(mat), lw=0.5)
    plt.plot('heatmap.png')


def plot_cnn_heatmap(item):
    """CNN heatmap"""
    pass


def pose_info(item):
    for k, v in item.items():
        if isinstance(v, np.ndarray):
            print(f'{k} shape: {v.shape}')
        else:
            print(f'{k}: {v}')


class Repcount:
    def __init__(self):
        self.anno_root = osp.join(config['proj_root'], 'datasets/RepCount/annotation')
        self.anno_train_path = osp.join(self.anno_root, 'train.csv')
        self.anno_val_path = osp.join(self.anno_root, 'val.csv')
        self.anno_test_path = osp.join(self.anno_root, 'test.csv')
        self.anno_all = None
        for split in ['train', 'val', 'test']:
            sp = self.get_anno(split)
            sp['split'] = split
            if self.anno_all is None:
                self.anno_all = sp
            else:
                self.anno_all = pd.concat([self.anno_all, sp])
        self.data_root = osp.join(config['proj_root'], 'data/RepCount/')
        self.pose_data = self.load_pose()

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

    def get_count(self, name):
        """Input: name, without .mp4
        Returns: list of frames in order of [start_1, end_1, start_2, end_2, ...]"""
        name = name+'.mp4'
        if name not in self.anno_all.name.unique():
            raise ValueError(f'Video {name} not found in annotation')
        isnum = self.anno_all[self.anno_all['name']
                              == name.strip()].iloc[0].dropna()
        count = isnum['count'].astype(int)
        reps = isnum.values[4:count*2+4].astype(int)
        return count, reps

    def get_video(self, name):
        """Input: name
        Returns: video_path"""
        video_dir = osp.join(self.data_root, 'videos')
        name = name+'.mp4'
        split = self.anno_all[self.anno_all['name'] == name.strip()]['split'].values[0]
        video_path = osp.join(video_dir, f'{split}/{name}')
        return video_path

    def load_pose(self):  # train set pose not extracted yet
        poses = np.load(osp.join(self.data_root, 'pose/val.pkl'), allow_pickle=True)
        d = {}
        for pose in poses:
            name = pose['frame_dir']  # without .mp4
            pose['count'], pose['reps'] = self.get_count(name)
            d[name] = pose
        return d

    def get_pose(self, name):
        """Input: name
        Returns: pose of the video"""
        if name not in self.pose_data:
            raise ValueError(f'Video {name} not found in pose data')
        return self.pose_data[name]

    def get_random_pose(self):
        """Returns: a random pose"""
        return self.pose_data[random.choice(list(self.pose_data.keys()))]

    def vis_pose(self, name):
        """Visualize pose of a video"""
        video_path = self.get_video(name)
        pass

    def __repr__(self):
        return f'Repcount(anno_root={self.anno_root})\n'\
            f'anno_train_path={self.anno_train_path}\n'\
            f'anno_val_path={self.anno_val_path},\n'\
            f'anno_test_path={self.anno_test_path}\n'\
            f'len_anno_all={len(self.anno_all)}\n'\
            f'columns={self.anno_all.columns[:10]}\n'


class Countix:
    def __init__(self):
        self.anno_root = osp.join(config['proj_root'], 'datasets/Countix')
        self.anno_train_path = osp.join(self.anno_root, 'workouts_train.csv')
        self.anno_val_path = osp.join(self.anno_root, 'workouts_val.csv')
        self.anno_all = None
        for split in ['train', 'val']:
            sp = self.get_anno(split)
            sp['split'] = split
            if self.anno_all is None:
                self.anno_all = sp
            else:
                self.anno_all = pd.concat([self.anno_all, sp])
        self.data_root = osp.join(config['proj_root'], 'data/Countix/')
        self.pose_data = self.load_pose()

    def get_anno(self, split='train'):
        split = split.lower()
        if split == 'train':
            anno = pd.read_csv(self.anno_train_path, delimiter=',')
        elif split == 'val':
            anno = pd.read_csv(self.anno_val_path, delimiter=',')
        elif split == 'test':
            raise ValueError(f'Invalid split: {split}. Test set not available yet')
        else:
            raise ValueError(f'Invalid split: {split}. Must be one of train, val')
        return anno

    def load_pose(self):
        train_pose = np.load(
            osp.join(self.data_root, 'pose/countix_train.pkl'),
            allow_pickle=True)
        val_pose = np.load(
            osp.join(self.data_root, 'pose/countix_val.pkl'),
            allow_pickle=True)
        d = {}
        for pose in train_pose:
            name = pose['frame_dir']
            pose['count'] = self.get_count(name)
            d[name] = pose
        for pose in val_pose:
            name = pose['frame_dir']
            pose['count'] = self.get_count(name)
            d[name] = pose
        return d

    def get_count(self, video_id):
        """Input: YouTube video_id
        Returns: number of repetitions"""
        if video_id not in self.anno_all.video_id.unique():
            raise ValueError(f'Video {video_id} not found in annotation')
        return self.anno_all[self.anno_all['video_id'] == video_id]['count'].values[0]

    def get_pose(self, video_id):
        """Input: YouTube video_id
        Returns: pose of the video"""
        if video_id not in self.pose_data:
            raise ValueError(f'Video {video_id} not found in pose data')
        return self.pose_data[video_id]

    def get_random_pose(self):
        return self.pose_data[random.choice(list(self.pose_data.keys()))]

    def get_video(self, video_id):
        """Input: YouTube video_id
        Returns: video_path"""
        video_dir = osp.join(self.data_root, 'videos')
        split = self.anno_all[self.anno_all['video_id'] == video_id].split.values[0]
        video_path = osp.join(video_dir, split, f'{video_id}.mp4')
        return video_path

    def __repr__(self):
        return f'Countix(anno_root={self.anno_root})\n'\
            f'anno_train_path={self.anno_train_path}\n'\
            f'anno_val_path={self.anno_val_path},\n'\
            f'len_anno_all={len(self.anno_all)}\n'\
            f'columns={self.anno_all.columns}\n'


if __name__ == '__main__':
    repcount = Repcount()
    # print(repcount)
    # countix = Countix()
    # item = countix.get_random_pose()
    item = repcount.get_random_pose()
    pose_info(item)
    # plot_time_series(item)
    # gen_gif(countix.get_video(item['frame_dir']), item)
    plot_pose_heatmap(item)
