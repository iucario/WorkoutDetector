# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import json
from random import choices
import time
import os
from collections import deque
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import collate, scatter

from mmaction.apis import init_recognizer
from mmaction.datasets.pipelines import Compose

from fastapi import Body, FastAPI, File, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import io
from PIL import Image
from base64 import b64decode

from inference import inference_video, test_pipeline


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

app = FastAPI()
origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        recv = await websocket.receive_text()
        if recv == 'close':
            await websocket.close()
            break
        elif recv == 'ping':
            await websocket.send_text('pong')
        else:
            recv = recv.split(',')[1]  # base64
            img = b64decode(recv)
            image = np.array(Image.open(io.BytesIO(img)))
            pred = await get_frame(image)
            await websocket.send_text(json.dumps(pred))


@app.post("/image")
async def read_image(payload: dict = Body(...)):
    image = payload['image']
    # print('blob:', image)
    if not image:
        return {'no image': 1}
    image = Image.open(io.BytesIO(b64decode(image.split(',', 1)[1])))
    # print(type(image), image.size)
    image = np.array(image)
    frame_queue.append(image)
    if data['img_shape'] is None:
        data['img_shape'] = image.shape[:2]
        print(f'img_shape: {data["img_shape"]}')
    if len(frame_queue) == sample_length:
        await real_time_inference(model, device)  # its blocking
    if result_queue:
        return result_queue.popleft()
    return {'no result': 1}


@app.post("/video")
async def read_video(video: bytes = File(...)):
    if not video:
        return {'no video': 1}
    with open('tmp.mp4', 'wb') as f:
        f.write(video)

    pred = main('tmp.mp4')
    return pred


global frame_queue, result_queue, data, pipeline, labels, sample_length
home = '~/projects/WorkoutDetector/mmaction2/'
config = os.path.join(
    home, 'configs/recognition/tsm/tsm_my_config.py')
checkpoint = '~/projects/WorkoutDetector/checkpoints/'\
    'tsm_r50_1x1x16_50e_sthv2_20220521.pth'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
cfg = Config.fromfile(config)
labels = [
    'front_raise', 'pull_up', 'squat', 'bench_pressing', 'jumping_jack',
    'situp', 'push_up', 'battle_rope', 'exercising_arm', 'lunge',
    'mountain_climber', ]
cfg.model.cls_head.num_classes = len(labels)
model = init_recognizer(cfg, checkpoint, device=device)

data = dict(img_shape=None, modality='RGB', label=-1)
data['num_clips'] = 16
data['clip_len'] = 1
sample_length = data['num_clips'] * data['clip_len']
pipeline = Compose(test_pipeline)
frame_queue = deque(maxlen=sample_length)
result_queue = deque(maxlen=1)

# example_home = 'example_videos/'
# example_videos = [os.path.join(example_home, x)
#                   for x in os.listdir(example_home)]

# video = 'example_videos/2jpteC44QKg.mp4'
# main(video)
