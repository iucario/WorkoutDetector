# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from random import choices
import time
import os
from collections import deque
from operator import itemgetter
from threading import Thread
import warnings
import cv2
import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.parallel import collate, scatter

from mmaction.apis import init_recognizer, inference_recognizer
from mmaction.datasets.pipelines import Compose
import gradio as gr

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
test_pipeline = [
    # dict(type='SampleFrames', clip_len=8, frame_interval=32, num_clips=1, test_mode=True),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]


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


def inference_video(model, video, device, cfg):
    print('Video:', video)
    capture = cv2.VideoCapture(video)
    if not capture.isOpened():
        print('Could not open video')
        return None

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    ret, frame = capture.read()

    frame_queue = []
    while ret:
        frame_queue.append(frame)
        ret, frame = capture.read()
    capture.release()
    print(f'video size[TWH]: {len(frame_queue)}x{width}x{height}')
    data = dict(img_shape=(height, width), modality='RGB',
                label=-1, start_index=0, total_frames=len(frame_queue))
    data['num_clips'] = 16
    data['clip_len'] = 1
    frame_queue = np.array(frame_queue)
    data['imgs'] = sample_frames(frame_queue, data['num_clips'])
    print(data['imgs'].shape)
    pipeline = Compose(test_pipeline)
    cur_data = pipeline(data)
    cur_data = collate([cur_data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        cur_data = scatter(cur_data, [device])[0]

    with torch.no_grad():
        scores = model(return_loss=False, **cur_data)[0]
        scores = list(enumerate(scores))
    num_selected_labels = min(len(scores), 5)
    scores.sort(key=lambda x: x[1], reverse=True)
    results = scores
    return results


def main(inputs):
    print('Inputs:', inputs)
    if not inputs:
        print('No video input')
        return None
    video = inputs
    results = inference_video(model, video, device, cfg)
    if not results:
        return None
    text = dict()
    for r in results:
        label = labels[r[0]]
        score = r[1]
        text[label] = float(score)
    print(text)
    return text


def dummy(image):
    return time.time()


async def real_time_inference(model, device):
    """Real-time inference"""
    print('Real-time inference')
    cur_windows = []
    if len(cur_windows) == 0:
        if len(frame_queue) == sample_length:
            cur_windows = list(np.array(frame_queue))

    cur_data = data.copy()
    cur_data['imgs'] = cur_windows
    cur_data = pipeline(cur_data)
    cur_data = collate([cur_data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        cur_data = scatter(cur_data, [device])[0]
    with torch.no_grad():
        scores = model(return_loss=False, **cur_data)[0]
        scores = list(enumerate(scores))
    num_selected_labels = min(len(scores), 5)
    scores.sort(key=lambda x: x[1], reverse=True)
    text = dict()
    for r in scores:
        label = labels[r[0]]
        score = r[1]
        text[label] = float(score)
    print(text)
    result_queue.append(text)


async def get_frame(frame):
    """Insert new frame to the queue and return result.
    frame: numpy array, HWC
    Returns: result of inference"""
    frame_queue.append(frame)
    if data['img_shape'] is None:
        data['img_shape'] = frame.shape[:2]
        print(f'img_shape: {data["img_shape"]}')
    if len(frame_queue) == sample_length:
        await real_time_inference(model, device)  # its blocking
    if result_queue:
        return result_queue.popleft()
    return {'no result': 1}


if __name__ == '__main__':
    global frame_queue, result_queue, data, pipeline, labels, sample_length
    home = 'mmaction2/'
    config = os.path.join(
        home,
        'configs/recognition/tsm/tsm_my_config.py')
    checkpoint = 'checkpoints/tsm_r50_256h_1x1x16_50e_sthv2_rgb_20220517.pth'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    cfg = Config.fromfile(config)
    labels = [
        'front_raise', 'pull_up', 'squat', 'bench_pressing', 'jumping_jack',
        'situp', 'push_up', 'others', 'battle_rope', 'pommelhorse']
    cfg.model.cls_head.num_classes = len(labels)
    model = init_recognizer(cfg, checkpoint, device=device)

    data = dict(img_shape=None, modality='RGB', label=-1)
    data['num_clips'] = 16
    data['clip_len'] = 1
    sample_length = data['num_clips'] * data['clip_len']
    pipeline = Compose(test_pipeline)
    frame_queue = deque(maxlen=sample_length)
    result_queue = deque(maxlen=1)

    demo = gr.Blocks()
    with demo:
        with gr.Row():
            inp = gr.Webcam(source='webcam', streaming=True, type="numpy")
            out = gr.Label(num_top_classes=5)
        btn = gr.Button("Run")
        btn.click(fn=get_frame, inputs=inp, outputs=out)
        

    demo = gr.Interface(
        fn=get_frame,
        inputs=[gr.Image(source='webcam', streaming=True,
                         type="numpy")],
        outputs="label",
        # examples=example_videos,
        title="MMAction2 webcam demo",
        description="Input a video file. Output the recognition result.",
        live=True,
        allow_flagging='never',
    )
    demo.launch()

    # example_home = 'example_videos/'
    # example_videos = [os.path.join(example_home, x)
    #                   for x in os.listdir(example_home)]

    # video = 'example_videos/2jpteC44QKg.mp4'
    # main(video)
