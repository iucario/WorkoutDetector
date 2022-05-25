import onnx
import onnxruntime

from collections import OrderedDict, deque
import time
import os
from typing import List, Tuple
import cv2
import numpy as np
import torch
from mmcv import Config
from torch import Tensor
from mmaction.apis import init_recognizer
from mmaction.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter

import warnings
warnings.filterwarnings('ignore')
onnxruntime.set_default_logger_severity(3)


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
test_pipeline = [
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
pipeline = Compose(test_pipeline)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
home = '~/projects/WorkoutDetector/mmaction2/'
config = os.path.join(
    home, 'configs/recognition/tsm/tsm_my_config.py')
checkpoint = '~/projects/WorkoutDetector/checkpoints/'\
    'tsm_r50_1x1x16_50e_sthv2_20220521.pth'

cfg = Config.fromfile(config)
labels = [
    'front_raise', 'pull_up', 'squat', 'bench_pressing', 'jumping_jack',
    'situp', 'push_up', 'battle_rope', 'exercising_arm', 'lunge',
    'mountain_climber', ]
# model = init_recognizer(cfg, checkpoint, device=device)

sample_length = 16
frame_queue = deque(maxlen=sample_length)
result_queue = deque(maxlen=1)

data = dict(img_shape=None, modality='RGB', label=-1)
data['num_clips'] = 16
data['clip_len'] = 1

onnx_ckpt = '../checkpoints/'\
    'tsm_r50_1x1x16_50e_sthv2_20220521.onnx'
onnx_model = onnx.load(onnx_ckpt)


def onnx_inference(inputs: Tensor) -> List[float]:
    """Inference with ONNX Runtime.
    Args: inputs: Tensor, shape=(1, 16, 3, 224, 224)
    """
    print('onnx_inference', 'inputs', inputs.shape)
    onnx.checker.check_model(onnx_model)
    # get onnx output
    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer = [
        node.name for node in onnx_model.graph.initializer
    ]
    net_feed_input = list(set(input_all) - set(input_initializer))
    assert len(net_feed_input) == 1
    sess = onnxruntime.InferenceSession(onnx_ckpt)
    onnx_scores = sess.run(
        None, {net_feed_input[0]: inputs.cpu().detach().numpy()})[0]
    # print(onnx_scores[0])
    onnx_scores = list(enumerate(onnx_scores[0]))
    onnx_scores.sort(key=lambda x: x[1], reverse=True)
    onnx_text = OrderedDict()
    for i, r in onnx_scores:
        label = labels[i]
        onnx_text[label] = float(r)
    print(onnx_text)
    return onnx_text


def dummy(image):
    return time.time()


async def real_time_inference(frame_queue, result_queue):
    """Real-time inference"""

    cur_windows = []
    if len(cur_windows) == 0:
        if len(frame_queue) == sample_length:
            if data['img_shape'] is None:
                data['img_shape'] = frame_queue[0].shape[:2]
                print(f'img_shape: {data["img_shape"]}')
            cur_windows = list(np.array(frame_queue))

    cur_data = data.copy()
    cur_data['imgs'] = cur_windows
    cur_data = pipeline(cur_data)
    # cur_data = collate([cur_data], samples_per_gpu=1)
    # if next(model.parameters()).is_cuda:
    #     cur_data = scatter(cur_data, [device])[0]
    # with torch.no_grad():
    #     scores = model(return_loss=False, **cur_data)[0]
    #     scores = list(enumerate(scores))
    onnx_scores = onnx_inference(torch.unsqueeze(cur_data['imgs'], 0))
    result_queue.append(onnx_scores)


async def get_frame(frame: Tensor):
    """Insert new frame to the queue and return result.
    frame: numpy array, HWC
    Returns: result of inference"""
    frame_queue.append(frame)

    if len(frame_queue) == sample_length:
        await real_time_inference(frame_queue, result_queue)
    if result_queue:
        return result_queue.popleft()
    return {'no result': 1}


def sample_frames(data, num):
    """Uniformly sample num frames from video, keep order"""
    total = len(data)
    if total <= num:
        # repeat last frame if num > total
        ret = np.vstack([data, np.repeat(data[-1], num-total,
                        axis=0).reshape(-1, *data.shape[1:])])
        return ret
    interval = total // num
    indices = np.arange(0, total, interval)[:num]
    for i, x in enumerate(indices):
        rand = np.random.randint(0, interval)
        if i == num - 1:
            upper = total
        else:
            upper = min(interval*(i+1), total)
        indices[i] = (x + rand) % upper
    assert len(indices) == num, f'len(indices)={len(indices)}'
    ret = data[indices]
    return ret


def inference_video(video):
    print('Video:', video)
    capture = cv2.VideoCapture(video)
    if not capture.isOpened():
        print('Could not open video')
        return None

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    ret, frame = capture.read()

    frames = []
    while ret:
        frames.append(frame)
        ret, frame = capture.read()
    capture.release()
    print(f'video size[TWH]: {len(frames)}x{width}x{height}')
    video_data = dict(img_shape=(height, width), modality='RGB',
                      label=-1, start_index=0, total_frames=len(frames))
    frames = np.array(frames)
    video_data['imgs'] = sample_frames(frames, sample_length)
    print(video_data['imgs'].shape)  # (16, W, H, 3)
    pipeline = Compose(test_pipeline)
    cur_data = pipeline(video_data)  # (16, 3, 224, 224)
    print(cur_data['imgs'].shape)
    # cur_data = collate([cur_data], samples_per_gpu=1)
    # if next(model.parameters()).is_cuda:
    #     cur_data = scatter(cur_data, [device])[0]
    # with torch.no_grad():
    #     srs = model(return_loss=False, **cur_data)[0]
    #     scores = list(enumerate(srs))
    onnx_scores = onnx_inference(torch.unsqueeze(cur_data['imgs'], 0))

    return onnx_scores


if __name__ == '__main__':

    inference_video('/home/umi/projects/WorkoutDetector/example_videos/4-YmQKoHYmw.mp4')
