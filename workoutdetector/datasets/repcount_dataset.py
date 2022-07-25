from dataclasses import dataclass
import os
import os.path as osp
from os.path import join as osj
from typing import List, Optional, Tuple, Dict
import einops
import torch
from torch import Tensor
import pandas as pd
import numpy as np
import base64
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
from torchvision.io import read_image
from workoutdetector.datasets.transform import sample_frames
from workoutdetector.settings import PROJ_ROOT, DATA_ROOT


def build_label_list(data_root: str,
                     anno_file: str,
                     actions: List[str],
                     out_dir: str,
                     overwrite: bool = False) -> None:
    """Build label list for common.ImageDataset
    
    Args:
        data_root (str): root directory of the RepCount dataset. Like `data/RepCount`.
        Will be used in RepcountHelper.
        anno_file (str): path to annotation file. Used in RepcountHelper.
        actions: list of str, action names. 
        Choose from ['situp', 'push_up', 'pull_up', 'jump_jack', 'squat', 'front_raise']
        out_dir (str): train.txt, val.txt, test.txt will be save in `out_dir`

    Example::

        >>> data_root = osj(PROJ_ROOT, 'data/RepCount')
        >>> actions = ['situp', 'push_up', 'pull_up', 'jump_jack', 'squat', 'front_raise']
        >>> out_dir = osp.join(PROJ_ROOT, 'data/RepImage')
        >>> build_label_list(data_root, REPCOUNT_ANNO_PATH, actions, out_dir, True)

        Creating directory: /work/data/RepImage
        Skip stu9_67.mp4 because count=0
        Done. First line in /work/data/RepImage/train.txt:
        rawframes/train/train951/img_00007.jpg 10
    """
    ACTIONS = ['situp', 'push_up', 'pull_up', 'jump_jack', 'squat', 'front_raise']
    for a in actions:
        assert a in ACTIONS, f'action {a} not in {ACTIONS}'
    train_txt, val_txt, test_txt = (osj(out_dir, 'train.txt'), osj(out_dir, 'val.txt'),
                                    osj(out_dir, 'test.txt'))
    if not osp.exists(out_dir):
        print('Creating directory:', out_dir)
        os.makedirs(out_dir)
    elif not overwrite:
        if osp.isfile(train_txt) or osp.isfile(val_txt) or osp.isfile(test_txt):
            print('Files already exist, will not overwrite')
            return
    filename_tmpl = "img_{:05d}.jpg"  # Index starts from 1
    helper = RepcountHelper(data_root=data_root, anno_file=anno_file)
    data = helper.get_rep_data(split=['train', 'val', 'test'], action=actions)
    train, val, test = open(train_txt, 'w'), open(val_txt, 'w'), open(test_txt, 'w')
    for item in data.values():
        if item.count < 1:
            print(f'Skip {item.video_name} because count={item.count}')
            continue
        start_idx, end_idx = item.reps[0], item.reps[1]  # Select the first rep
        mid_idx = (start_idx + end_idx) // 2
        start_img = filename_tmpl.format(start_idx + 1)
        mid_img = filename_tmpl.format(mid_idx + 1)
        cls_idx = actions.index(item.class_)
        rel_path = osp.relpath(item.frames_path, data_root)
        if item['split'] == 'train':
            train.write(f"{rel_path}/{start_img} {cls_idx*2}\n")
            train.write(f"{rel_path}/{mid_img} {cls_idx*2+1}\n")
        elif item['split'] == 'val':
            val.write(f"{rel_path}/{start_img} {cls_idx*2}\n")
            val.write(f"{rel_path}/{mid_img} {cls_idx*2+1}\n")
        elif item['split'] == 'test':
            test.write(f"{rel_path}/{start_img} {cls_idx*2}\n")
            test.write(f"{rel_path}/{mid_img} {cls_idx*2+1}\n")
    train.close()
    val.close()
    test.close()
    with open(train_txt) as f:
        line = f.readline()
    print(f'Done. First line in {train_txt}:\n{line}')


def parse_onedrive(link: str) -> str:
    """Parse onedrive link to download link.
    
    Args:
        link: str, start with `https://1drv.ms/u/s!`
    Returns:
        str, download link.
    """
    assert link.startswith('https://1drv.ms/u/s!')
    b = base64.urlsafe_b64encode(link.strip().encode('ascii'))
    s = b.decode('ascii')
    res = f'https://api.onedrive.com/v1.0/shares/u!{s}/root/content'
    return res


def eval_count(preds: List[int], targets: List[int]) -> Tuple[float, float]:
    """Evaluate count prediction. By mean absolute error and off-by-one error."""

    mae = 0.0
    off_by_one = 0.0
    for pred, target in zip(preds, targets):
        mae += abs(pred - target)
        off_by_one += (abs(pred - target) == 1)
    return mae / len(preds), off_by_one / len(preds)


@dataclass
class RepcountItem:
    """RepCount dataset video item"""

    video_path: str  # the absolute video path
    frames_path: str  # the absolute rawframes path
    total_frames: int
    class_: str
    count: int
    reps: List[int]  # start_1, end_1, start_2, end_2, ...
    split: str
    video_name: str
    fps: float = 30.0
    ytb_id: Optional[str] = None  # YouTube id
    ytb_start_sec: Optional[int] = None  # YouTube start sec
    ytb_end_sec: Optional[int] = None  # YouTube end sec

    def __str__(self):
        return (f'video: {self.video_name}\nclass: {self.class_}\n'
            f'count: {self.count}\nreps: {self.reps}\nfps: {self.fps}')

    def __getitem__(self, key):
        return self.__dict__[key]

    def __iter__(self):
        return iter(self.__dict__.items())


@dataclass
class RepcountItemWithPred(RepcountItem):
    """RepCount dataset video item with prediction"""

    pred_count: int = 0
    pred_reps: Optional[List[int]] = None
    mae: float = 0  # mean absolute error
    obo_acc: bool = False  # pred is correct if difference within 1
    model_type: Optional[str] = None  # model type. image or video


class RepcountHelper:
    """Helper class for RepCount dataset
    Extracting annotations, evaluation and helpful functions
    
    Args:
        data_root: the data root path, e.g. 'data/RepCount'
        ann_file: the annotation file path
    """

    def __init__(self, data_root: str, anno_file: str):

        self.anno_file = anno_file
        self.data_root = data_root
        self.classes = [
            'situp', 'push_up', 'pull_up', 'jump_jack', 'squat', 'front_raise'
        ]  # no bench_pressing because data not cleaned yet.

    def get_rep_data(self,
                     split: List[str] = ['test'],
                     action: List[str] = ['situp']) -> Dict[str, RepcountItem]:
        """
        Args:
            split (List[str]): list of the split names
            action (List[str]): list of the action names. If ['all'], all actions are used.

        Returns:
            dict, name: RepcountItem
        """
        assert len(split) > 0, 'split must be specified, e.g. ["train", "val"]'
        assert len(action) > 0, 'action must be specified, e.g. ["pull_up", "squat"]'
        split = [x.lower() for x in split]
        action = [x.lower() for x in action]
        if 'all' in action:
            action = self.classes
        df = pd.read_csv(self.anno_file, index_col=0)
        df = df[df['split'].isin(split)]
        df = df[df['class_'].isin(action)]
        df = df.reset_index(drop=True)
        ret = {}
        for idx, row in df.iterrows():
            name = row['name']
            name_no_ext = name.split('.')[0]
            class_ = row['class_']
            split_ = row['split']
            video_path = os.path.join(self.data_root, 'videos', split_, name)
            frame_path = os.path.join(self.data_root, 'rawframes', split_, name_no_ext)
            total_frames = -1
            if os.path.isdir(frame_path): # TODO: this relies on rawframe dir. Not good.
                total_frames = len(os.listdir(frame_path))
            video_id = row['vid']
            count = int(row['count'])
            if count > 0:
                reps = [int(x) for x in row['reps'].split()]
            else:
                reps = []
            item = RepcountItem(video_path, frame_path, total_frames, class_, count, reps,
                                split_, name, row.fps, video_id, row['start'], row['end'])
            ret[name] = item
        return ret

    def eval_count(
            self,
            pred_reps: Dict[str, int],
            split: List[str] = ['test'],
            action: List[str] = []
    ) -> Tuple[float, float, Dict[str, RepcountItemWithPred]]:
        """Evaluate repetition count prediction
        
        Args:
            pred_reps: dict, name: count
            action: list of the action names
            split: list of the split names

        Returns:
            tuple, (mean_avg_error, off_by_one_acc)
        
        TODO:
            - add metrics for precise repetition timestamp. Use heatmap to smooth the error
        """

        items = self.get_rep_data(split=split, action=action)
        total_mae = 0.0
        total_off_by_one = 0.0
        pred_items: Dict[str, RepcountItemWithPred] = {}
        for name, count in pred_reps.items():
            gt_count = items[name].count
            diff = abs(count - gt_count)
            if gt_count > 0:
                mae = diff / gt_count
            else:
                mae = 0  # Not decided how to handle 0 repetition case
            obo_acc = (diff <= 1)
            total_mae += mae
            total_off_by_one += obo_acc
            pred_items[name] = RepcountItemWithPred(**items[name].__dict__,
                                                    pred_count=count,
                                                    pred_reps=[],
                                                    mae=mae,
                                                    obo_acc=obo_acc)
        return total_mae / len(items), total_off_by_one / len(items), pred_items


class RepcountDataset(torch.utils.data.Dataset):  # type: ignore
    """Repcount dataset
    https://github.com/SvipRepetitionCounting/TransRAC
    
    Args:
        root: str, root dir
        split: str, train or val or test
    
    Properties:
        classes: list of str, class names
        df: pandas.DataFrame, annotation data
        split: str, train or val or test
        transform: callable, transform for rawframes
    
    Notes:
        File tree::

            |- RepCount
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

        The annotation csv file is expected to be in:
        `{PROJ_ROOT}/datasets/RepCount/annotations.csv`

    """
    _URL_VIDEO = 'https://1drv.ms/u/s!AiohV3HRf-34ipk0i1y2P1txpKYXFw'
    _URL_ANNO = 'https://1drv.ms/f/s!AiohV3HRf-34i_V9MWtdu66tCT2pGQ'
    _URL_RAWFRAME = 'https://1drv.ms/u/s!AiohV3HRf-34ipwACYfKSHhkZzebrQ'

    def __init__(self,
                 root: str,
                 split: str = 'train',
                 transform=None,
                 download=False) -> None:
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
        anno_path = os.path.join(PROJ_ROOT, 'datasets/RepCount/annotation.csv')
        if not os.path.exists(anno_path):
            raise OSError(f'{anno_path} not found.')
        self._anno_df = pd.read_csv(anno_path, index_col=0)
        self.df = self._anno_df[self._anno_df['split'] == split]
        self.classes = self.df['class_'].unique().tolist()
        self.transform = transform

    def get_video(self, index: int) -> Tuple[str, int]:
        """Returns path to video rawframe and video class.
        For action recognition.
        """

        row = self.df.iloc[index]
        video_frame_path = os.path.join('RepCount/rawframes', row['split'], row['name'])
        label = self.classes.index(row['class'])
        count = row['count']
        reps = list(map(int, row.reps.split())) if count else []
        return video_frame_path, label

    def get_video_list(self,
                       split: str,
                       action: Optional[str] = None,
                       max_reps: int = 2) -> List[dict]:
        """

        Args:
            split: str, train or val or test
            action: str, action class name. If none, all actions are used.
            max_reps: int, limit the number of repetitions per video.
                If less than 1, all repetitions are used.

        Returns:
            list of dict: videos, 
                {
                    video_path: path to raw frames dir, relative to `root`
                    start: start_frame_index, start from 1,
                    end: end_frame_index
                    length: end_frame_index - start_frame_index + 1
                    class: action class,
                    label: 0 or 1
                }
        """
        df = self._anno_df[self._anno_df['split'] == split]
        if action is not None:
            df = df[df['class_'] == action]
        videos = []
        for row in df.itertuples():
            name = row.name.split('.')[0]
            count = row.count
            if count > 0:
                reps = list(map(int, row.reps.split()))[:max_reps * 2]
                for start, end in zip(reps[0::2], reps[1::2]):
                    start += 1  # plus 1 because img index starts from 1
                    end += 1  # but annotated frame index starts from 0
                    mid = (start + end) // 2
                    videos.append({
                        'video_path': os.path.join('RepCount/rawframes', split, name),
                        'start': start,
                        'end': mid,
                        'length': mid - start + 1,
                        'class': row.class_,
                        'label': 0
                    })
                    videos.append({
                        'video_path': os.path.join('RepCount/rawframes', split, name),
                        'start': mid + 1,
                        'end': end,
                        'length': end - mid,
                        'class': row.class_,
                        'label': 1
                    })
        return videos

    def __len__(self) -> int:
        return len(self.df)

    def _download(self) -> None:
        """
        Download the RepCount dataset archive from OneDrive and extract it under root.
        """
        if self._check_exists():
            return
        # the extracted folder is `rawframes`, may upload again sometime
        url = parse_onedrive(self._URL_RAWFRAME)
        download_and_extract_archive(url,
                                     download_root=self._data_path,
                                     filename='rawframes.zip',
                                     extract_root=self._data_path)

    def _check_exists(self) -> bool:
        return os.path.exists(self._data_path) and os.path.isdir(
            self._data_path) and os.path.exists(
                os.path.join(self._data_path, 'RepCount/rawframes'))


class RepcountImageDataset(RepcountDataset):
    """Repcount image dataset for binary classification. 
    Start and end state in specific action.

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

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
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
        root: str, data root, e.g. './data'
        action: str
        num_frames: int, number of frames in one video
    
    Properties:
        video_list: list of dict, each dict contains:
        ::
        
            video_path: path_to_raw_frames_dir, 
            start: start_frame_index, start from 1,
            end: end_frame_index
            length: end_frame_index - start_frame_index + 1
            class: action class,
            label: 0 or 1
    
    Returns:
        Tensor, shape (N, C, H, W)
        List[int], label
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
        self.num_segments = num_segments
        self.video_list = self.get_video_list(split, action)

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
        frame_tensor = torch.stack(frame_list, 0)
        assert frame_tensor.shape[0] == self.num_segments, \
            f'frame_list.shape[0] = {frame_tensor.shape[0]}, ' \
            f'but self.num_segments = {self.num_segments}'
        return frame_tensor, self.video_list[index]['label']

    def __len__(self) -> int:
        return len(self.video_list)


class RepcountRecognitionDataset(torch.utils.data.Dataset):
    """RepCount action recognition(video classification) dataset
    
    Args:
        root (str): data root, e.g. './data/RepCount'
        split (str): 'train' or 'val'
        transform: torch.transforms.Compose, transform for image
        num_segments (int): number of frames in one video
        download: bool, whether to download the dataset
    """

    def __init__(self,
                 root: str,
                 split: str,
                 actions: Optional[List[str]] = None,
                 num_segments: int = 8,
                 transform=None) -> None:
        super(RepcountRecognitionDataset, self).__init__()
        self.split = split
        self.transform = transform
        assert os.path.isdir(root), f'{root} does not exist'
        anno_path = os.path.join(root, 'annotation.csv')
        assert os.path.isfile(anno_path), f'{anno_path} does not exist'
        helper = RepcountHelper(data_root=root, anno_file=anno_path)
        if actions is None:
            actions = helper.classes
        video_dict = helper.get_rep_data(split=[split], action=actions)
        self.video_list: List[RepcountItem] = list(video_dict.values())
        self.num_seg = num_segments
        self.action_map = dict(zip(actions, range(len(actions))))

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        video = self.video_list[index]
        try:
            rep_start, rep_end = video.reps[0], video.reps[-1]
        except IndexError:  # no reps
            rep_start, rep_end = 0, video.total_frames - 1
        idx_list = sample_frames(rep_end - rep_start, self.num_seg, offset=rep_start)
        frame_list = [
            read_image(os.path.join(video.frames_path, f'img_{i+1:05}.jpg'))
            for i in idx_list
        ]
        if self.transform:
            frame_list = [
                self.transform(
                    read_image(os.path.join(video.frames_path, f'img_{i+1:05}.jpg')))
                for i in idx_list
            ]
        label = self.action_map[video.class_]
        frame_tensor = torch.stack(frame_list, 0)
        frame_tensor = frame_tensor.permute(1, 0, 2, 3)
        return frame_tensor, label

    def __len__(self) -> int:
        return len(self.video_list)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    PROJ_ROOT = os.path.expanduser('~/projects/WorkoutDetector/')
    data_root = os.path.join(PROJ_ROOT, 'data')
    dataset = RepcountVideoDataset(data_root,
                                   split='test',
                                   action='jump_jack',
                                   num_segments=8)
    # print(dataset.classes)
    random_index = np.random.randint(0, len(dataset))
    img, label = dataset[random_index]
    plt.figure(figsize=(8, 4), dpi=200)
    img = einops.rearrange(img, '(b1 b2) c h w -> (b1 h) (b2 w) c', b2=4)
    plt.title(f'label: {label}')
    print(img.shape)
    plt.imshow(img)
    plt.show()