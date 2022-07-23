import asyncio
import os
import time
import warnings
from collections import Counter, OrderedDict, defaultdict, deque
from typing import Deque, Dict, List, Tuple

import cv2
import numpy as np
import onnx
import onnxruntime
import torch
import torchvision.transforms as T
from fastapi import WebSocket
from PIL import Image
from torch import Tensor
from torchvision.io import read_video
from workoutdetector.trainer import LitModel
from workoutdetector.utils import pred_to_count

warnings.filterwarnings('ignore')
onnxruntime.set_default_logger_severity(3)

data_transform = T.Compose([
    # T.ToPILImage(),
    T.ConvertImageDtype(dtype=torch.float32),
    T.Resize(256),
    T.CenterCrop(224),
    # T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
sample_length = 8

action_11_labels = [
    'front_raise',
    'pull_up',
    'squat',
    'bench_pressing',
    'jumping_jack',
    'situp',
    'push_up',
    'battle_rope',
    'exercising_arm',
    'lunge',
    'mountain_climber',
]

rep_6_labels = ['situp', 'push_up', 'pull_up', 'jump_jack', 'squat', 'front_raise']
PROJ_ROOT = os.path.expanduser('~/projects/WorkoutDetector')
onnx_ckpt = os.path.join(PROJ_ROOT, 'checkpoints/action-11/tsm_1x1x8_sthv2_20220522.onnx')
onnx_sess = onnxruntime.InferenceSession(onnx_ckpt, providers=['CUDAExecutionProvider'])
ckpt_path = os.path.join(
    PROJ_ROOT, 'checkpoints/repcount-12/rural-river-23-repcount-12-20220711-191616.pt')
# torch_model = LitModel.load_from_checkpoint(
#     'checkpoints/repcount-12/best-val-acc=0.841-epoch=26-20220711-191616.ckpt')
torch_model = torch.jit.load(ckpt_path)
torch_model.eval()


def onnx_inference(model, inputs: Tensor) -> List[Tuple[int, float]]:
    """Inference with ONNX Runtime.

    Args: 
        inputs (Tensor): shape=(8, 3, 224, 224)
    Returns:
        List[Tuple[int, float]]: List of (label, score)
    """
    # print('onnx_inference', 'inputs', inputs.shape)
    inputs = torch.unsqueeze(inputs, 0)
    input_name = model.get_inputs()[0].name
    ort_outs = model.run(None, {input_name: inputs.cpu().numpy()})
    score: np.ndarray = ort_outs[0][0]
    pred = list(enumerate(score.tolist()))
    pred.sort(key=lambda inputs: inputs[1], reverse=True)
    return pred


def torch_inference(model, inputs: Tensor) -> List[Tuple[int, float]]:
    outputs = model(inputs.to(device))[0]
    outputs = outputs.softmax(dim=0)
    pred = list(enumerate(outputs.cpu().detach().numpy()))
    pred.sort(key=lambda inputs: inputs[1], reverse=True)
    return pred


def inference_video(model, inputs):
    if isinstance(inputs, np.ndarray):
        inputs = torch.from_numpy(inputs)
    # print(inputs.shape)
    return torch_inference(model, data_transform(inputs))


def count_rep_video(video_path) -> Tuple[int, List[int], List[str]]:
    model = torch_model
    threshold = 0.5
    assert os.path.exists(video_path), f'{video_path} not exists'
    input_queue: List[Tensor] = []
    count = 0
    states: List[int] = []  # onnx preds
    frame_idx = 0
    vid = read_video(video_path)[0].permute(0, 3, 1, 2)

    for i in range(0, vid.shape[0] - 8, 8):
        input_clip = vid[i:i + sample_length]
        assert input_clip.shape[:2] == (8, 3), input_clip.shape
        pred = inference_video(model, input_clip)
        assert abs(sum(x[1] for x in pred) - 1) <= 1e-3, pred
        pred_class = pred[0][0] if pred[0][1] > threshold else -1
        states.append(pred_class)
        print(pred[0])

    count, reps = pred_to_count(states, stride=8, step=1)
    actions = [rep_6_labels[states[r // 8] // 2] for r in reps[::2]]
    return count, reps, actions


async def dummy(image):
    print('func dummy', image.shape)
    time.sleep(1)
    return {'dummy': 1}


async def get_frame(websocket: WebSocket, queue: asyncio.Queue) -> None:
    """Insert new frame to the queue and return result.

    Args:
        websocket (WebSocket): WS connection
        queue (asyncio.Queue): queue to put new frame

    Result JSON format::

        {
            success: bool,
            msg: str,
            type: str, 'rep' or 'action',
            data: {
                score: {
                    action_1: float # softmax score of action_1
                    ...
                },
                count: {
                    action_1: int, # count of action_1
                    action_2: int,
                    ...
                }
            }
        }
    """
    count = 0
    states: List[int] = []
    frame_queue = []
    action_count: Dict[str, int] = defaultdict(int)
    while True:
        frame = await queue.get()
        print(frame.shape)
        frame_queue.append(frame)
        if len(frame_queue) == sample_length:
            input_clip = np.array(frame_queue)  # [8, 3, 224, 224]
            pred = inference_video(torch_model, input_clip)
            action_score = dict([(rep_6_labels[x[0] // 2], x[1]) for x in pred])
            pred_class = pred[0][0] if pred[0][1] > 0.5 else -1
            pred_action = rep_6_labels[pred_class // 2]
            # check previous 3 states whether state changed
            if pred_class % 2 == 1:  # if state 1
                for i in range(-3, 0):
                    # if previous state
                    if states[i] // 2 == pred_class // 2 and states[i] % 2 == 0:
                        count += 1
                        action_count[pred_action] += 1
                        break
            states.append(pred_class)
            print(f'{pred_class} {count}')
            if action_score:
                print(action_score)
                msg = dict(success=True,
                           msg='success',
                           type='rep',
                           data=dict(score=action_score, count=dict(count=action_count)))
                await websocket.send_json(msg)
            frame_queue.clear()


if __name__ == '__main__':

    video = '/home/umi/data/RepCount/videos/train/test1465.mp4'
    count, reps, actions = count_rep_video(video)
    print(f'count={count}, reps={reps}')
    print(f'actions={actions}')
