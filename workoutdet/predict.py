import argparse
import json
import os
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torchvision.transforms as T
from einops import rearrange
from torch import Tensor, nn
from torchvision.io import read_video

from workoutdet.data import get_rep_data
from workoutdet.trainer import LitModel


def pred_to_count(preds: List[int], stride: int, step: int = 1) -> Tuple[int, List[int]]:
    """Convert a list of predictions to a repetition count.
    
    Args:
        preds (List[int]): list of size total_frames//stride in the video. If -1, it means no action.
        stride (int): predict every stride frames.
        step (int): step size of the sampled frames. Does not effect the result.

    Returns:
        A tuple of (repetition count, list of preds of action start and end states, 
            e.g. start_1, end_1, start_2, end_2, ...)

    Note:
        The labels are in order. Because that's how I loaded the data.
        E.g. 0 and 1 represent the start and end of the same action.
        We consider action class as well as state changes.
        I can ensemble a standalone action recognition model if things don't work well.

    Algorithm:
        1. If the state changes, and current and previous state are of the same action,
            and are in order, we count the action. For example, if the state changes from
            0 to 1, or 2 to 3, aka even to odd, we count the action.
        
        It means the model has to capture the presice time of state transition.
        Because the model takes 8 continuous frames as input.
        Or I doubt it would work well. Multiple time scale should be added.
    
    Example:
        >>> preds = [-1, -1, 6, 6, 6, 7, 6, 6, 6, 7, 6, 6, 7, 7, 6, 6, 7, 7, 6, 6, 7, 7, 6, 6, 7, 7, -1]
        >>> pred_to_count(preds, stride=8, step=2)
        (6, [16, 40, 48, 72, 80, 96, 112, 128, 144, 160, 176, 192])
    """

    count = 0
    reps = []  # start_1, end_1, start_2, end_2, ...
    states: List[int] = []
    prev_state_start_idx = 0
    for idx, pred in enumerate(preds):
        if pred == -1:
            continue
        # if state changes and current and previous state are the same action
        if states and states[-1] != pred:
            if pred % 2 == 1 and states[-1] == pred - 1:
                count += 1
                reps.append(prev_state_start_idx * stride)
                reps.append(idx * stride)
        states.append(pred)
        prev_state = preds[prev_state_start_idx]
        if pred != prev_state:  # new state, new start index
            prev_state_start_idx = idx

    assert count * 2 == len(reps)
    return count, reps  # len(rep) * step <= len(frames), last not full queue is discarded


class TestDataset(torch.utils.data.Dataset):
    """DataLoader for one video
    
    Args:
        video_path: path to video
        stride (int): stride of prediction frame indices. Default 1.
        step (int): step of sampling frames. Default 1.
        length (int): length of input frames to model. Default 8.
        input_shape (str): shape of input frames. Default 'TCHW'. 'CTHW' for 3D CNN.
        transform (Callable): transform for frames.
    """

    def __init__(self,
                 video_path: str,
                 stride: int = 1,
                 step: int = 1,
                 length: int = 8,
                 input_shape: str = 'TCHW',
                 transform: Optional[Callable] = None):
        assert step >= 1, 'step must be greater than or equal to 1'
        assert stride >= 1, 'stride must be greater than or equal to 1'

        video, _, meta = read_video(video_path)
        # start index of each inputs.
        self.indices = list(range(0, len(video) - step * length + 1, stride))
        self.shape = ' '.join(list(input_shape))
        self.video = video.permute(0, 3, 1, 2)  # TCHW
        self.meta = meta
        self.fps: float = meta['video_fps']
        self.step = step
        self.length = length
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        """Get a frame and its start frame index."""

        i = self.indices[index]
        frames = self.video[i:i + self.step * self.length:self.step]
        if self.transform is not None:
            frames = self.transform(frames)
        t, c, h, w = frames.shape
        frames = rearrange(frames, f'T C H W -> {self.shape}', T=t, C=c, H=h, W=w)
        return frames, i

    def __len__(self) -> int:
        return len(self.indices)


@torch.no_grad()
def predict_one_video(model, path: str, out_path: str, stride: int, step: int, transform,
                      total_frames: int, reps: List[int], class_name: str, device: str):
    """Predict one video and save scores to JSON file."""
    ds = TestDataset(path, stride=stride, step=step, length=8, transform=transform)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
    video_name = path.split('.')[0] + '.mp4'
    res_dict = dict(
        video_name=video_name,
        model='TSM',
        stride=stride,
        step=step,
        length=8,
        fps=ds.fps,
        input_shape=[1, 8, 3, 224, 224],
        checkpoint=ckpt,
        total_frames=total_frames,
        ground_truth=reps,
        action=class_name,
    )
    scores: Dict[int, dict] = dict()
    for x, i in loader:
        start_index = i.item()
        with torch.no_grad():
            pred: Tensor = model(x.to(device))
            scores[start_index] = dict((j, v.item()) for j, v in enumerate(pred[0]))
        # print(scores[start_index])
    res_dict['scores'] = scores

    json.dump(res_dict, open(out_path, 'w'))
    print(f'{video_name} result saved to {out_path}')


def main(ckpt: str,
         out_dir: str,
         stride: int = 1,
         step: int = 1,
         rank: int = 0,
         world_size: int = 1):
    """Inference videos in the dataset and save results to JSON"""

    device = f'cuda:{rank}'
    data_root = os.path.expanduser('~/data/RepCount')
    data = get_rep_data(anno_path=os.path.join(data_root, 'annotation.csv'),
                        data_root=data_root,
                        split=['train', 'val', 'test'],
                        action=['all'])
    # data parallel
    part_size = len(data) // world_size
    if rank == world_size - 1:
        end = len(data)
    else:
        end = part_size * (rank + 1)
    transform = T.Compose([
        T.ConvertImageDtype(torch.float32),
        T.Resize(256),
        T.CenterCrop((224, 224)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    model = LitModel.load_from_checkpoint(ckpt)
    model.to(device)
    model.eval()
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    for item in list(data.values())[rank * part_size:end]:
        out_path = os.path.join(out_dir,
                                f'{item.video_name}.stride_{stride}_step_{step}.json')
        if os.path.exists(out_path):
            print(f'{out_path} already exists. Skip.')
            continue
        predict_one_video(model, item.video_path, out_path, stride, step, transform,
                          item.total_frames, item.reps, item.class_, device)


if __name__ == '__main__':
    ckpt = 'checkpoints/repcount-12/best-val-acc=0.841-epoch=26-20220711-191616.ckpt'
    out_dir = f'out/acc_0.841_epoch_26_20220711-191616_1x1'
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default=ckpt)
    parser.add_argument('--out_dir', type=str, default=out_dir)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    args = parser.parse_args()

    main(ckpt,
         out_dir,
         stride=args.stride,
         step=1,
         rank=args.rank,
         world_size=args.world_size)
