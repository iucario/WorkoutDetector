import random
import sys
import os
import os.path as osp
from typing import List
import PIL
import cv2
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import yaml
import sklearn.metrics.pairwise as pw
import seaborn as sns
import torch
from torch import Tensor, nn
import torchvision.transforms as T
import timm

from WorkoutDetector.utils.visualize import Vis2DPose

config = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), 'config.yml')))

KPS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder',
    'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
    'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]
CLASSES = [
    'front_raise', 'pull_up', 'squat', 'bench_pressing', 'jumping_jack', 'situp',
    'push_up', 'battle_rope', 'exercising_arm', 'lunge', 'mountain_climber'
]


def gen_gif(video_path, anno):
    """Generate 2D Skeletons with the original RGB Video to `{vid}.gif`"""

    frame_dir = anno['frame_dir']
    vid = Vis2DPose(anno,
                    thre=0.5,
                    out_shape=(540, 960),
                    layout='coco',
                    fps=20,
                    video=video_path)
    vid.to_gif(f'{frame_dir}.gif')


def plot_time_series(item):
    """Plot time series of a single pose item

    Args:
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
    plt.title(name + str(item['count']))
    plt.show()


def plot_pose_heatmap(item):
    """Cosine similarity between 2D poses heatmap
        item: Pose item
        reps: list of repetition starts and ends"""
    kp = item['keypoint'][0]
    num_frame = kp.shape[0]
    mat = np.zeros((num_frame, num_frame))
    feat = kp.reshape(num_frame, -1)
    mat = pw.pairwise_distances(feat, metric='cosine')
    sns.heatmap(mat, cmap='viridis')
    plt.title(f"pose {item['frame_dir']} {item['count']} reps")
    if 'reps' in item:
        plt.vlines(item['reps'][::2], colors='r', ymin=0, ymax=len(mat), lw=0.5)
    plt.plot('heatmap.png')


def cnn_feature(fx, imgs: List, device='cuda') -> Tensor:
    """CNN feature extractor

    Args:
        fx: CNN model
        imgs: list of PIL.Image
        device: str, 'cuda' or 'cpu'

    Returns:
        Tensor of shape (num_img, feature_dim)
    """

    batch_size = 10
    fx.to(device)
    transforms = T.Compose([
        T.ToTensor(),
        T.Resize(224),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    imgs = [transforms(img) for img in imgs]
    imgs = torch.stack(imgs)
    imgs = imgs.to(device)
    features = torch.zeros((len(imgs), 512))
    with torch.no_grad():
        for i in range(0, len(imgs), batch_size):
            o = fx(imgs[i:i + batch_size])
            features[i:i + batch_size] = o.cpu()
    return features


def video_feature(timm_model, video_path):
    """Video feature extractor

    Args:
        timm_model: timm model name. e.g. 'resnet50'
        video_path: str, path to video

    Returns:
        Tensor of shape (num_frame, feature_dim)
    """

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    frames = []
    while ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(PIL.Image.fromarray(frame))
        ret, frame = cap.read()
    if not frames:
        return
    fx = timm.create_model(timm_model, pretrained=True, num_classes=0)
    return cnn_feature(fx, frames)


def plot_sim(feats: Tensor, count=0, reps=None):
    sim = pw.pairwise_distances(feats.numpy(), metric='cosine')
    # row-wise softmax. Repnet did this.
    # maxes = np.tile(np.max(sim, axis=1),(sim.shape[0], 1))
    # sim = np.exp(sim-maxes)/(np.sum(np.exp(sim-maxes), axis=1, keepdims=True))
    sns.heatmap(sim, cmap='viridis')
    plt.title(f'CNN {count} reps')
    if reps is not None:
        plt.vlines(reps[0::2], colors='r', ymin=0, ymax=len(sim), lw=0.3)
        plt.vlines(reps[1::2], colors='g', ymin=0, ymax=len(sim), lw=0.3)
    plt.show()


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
        """Returns annotation dataframe

        Args:
            split: str, 'train', 'val', 'test'

        Returns:
            dataframe
        """

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

    def get_count(self, name: str):
        """Returns count of repetitions and repetition starts and ends frames

        Args:
            name: str, name of video

        Returns: 
            int: number of repetitions.
            list of frames in order of [start_1, end_1, start_2, end_2, ...]
        """

        if not name.endswith('.mp4'):
            name = name + '.mp4'
        if name not in self.anno_all.name.unique():
            raise ValueError(f'Video {name} not found in annotation')
        isnum = self.anno_all[self.anno_all['name'] == name.strip()].iloc[0].dropna()
        count = isnum['count'].astype(int)
        reps = isnum.values[4:count * 2 + 4].astype(int)
        return count, reps

    def get_video(self, name: str):
        """Returns video path from name

        Args:
            name: str, name of video

        Returns: 
            str: video_path
        """

        video_dir = osp.join(self.data_root, 'videos')
        name = name + '.mp4' if not name.endswith('.mp4') else name
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

    def get_pose(self, name: str):
        """Returns pose data for video

        Args:
            name: str, name of video

        Returns: 
            dict: pose of the video. Same format as OpenMMLab's.
        """
        if name not in self.pose_data:
            raise ValueError(f'Video {name} not found in pose data')
        return self.pose_data[name]

    def get_random_pose(self):
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
        train_pose = np.load(osp.join(self.data_root, 'pose/countix_train.pkl'),
                             allow_pickle=True)
        val_pose = np.load(osp.join(self.data_root, 'pose/countix_val.pkl'),
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
        """Returns count of repetitions

        Args:
            video_id: str, name of video

        Returns:
            int: number of repetitions
        """

        if video_id not in self.anno_all.video_id.unique():
            raise ValueError(f'Video {video_id} not found in annotation')
        return self.anno_all[self.anno_all['video_id'] == video_id]['count'].values[0]

    def get_pose(self, video_id: str):
        """Returns extracted pose of a video

        Args:
            video_id: str, YouTube id

        Returns:
            dict: containing the pose info of the video
        """

        if video_id not in self.pose_data:
            raise ValueError(f'Video {video_id} not found in pose data')
        return self.pose_data[video_id]

    def get_random_pose(self):
        return self.pose_data[random.choice(list(self.pose_data.keys()))]

    def get_video(self, video_id):
        """Returns video path from video id
        
        Args:
            video_id: str, YouTube id
        
        Returns:
            str: video_path
        """

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
    df = repcount.get_anno('test')
    df = df[df['type'] == 'squat']
    names = df['name'].values
    print(names)
    for name in names:
        print(repcount.get_video(name))
