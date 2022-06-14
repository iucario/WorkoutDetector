import onnx
import onnxruntime

from collections import OrderedDict
import os
from typing import List, Tuple
import cv2
import numpy as np
import torch
from mmcv import Config
from torch import Tensor
from mmaction.datasets.pipelines import Compose
from mmaction.apis import init_recognizer
from mmcv.parallel import collate, scatter

import tempfile

import gradio as gr
import warnings
import yaml

user_cfg = yaml.safe_load(
    open(os.path.join(os.path.dirname(__file__), 'utils/config.yml')))

warnings.filterwarnings('ignore')
onnxruntime.set_default_logger_severity(3)

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_bgr=False)
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
home = user_cfg['proj_root']
print('HOME', home)
config = os.path.join(home, 'mmaction2/configs/recognition/tsm/tsm_my_config.py')
sample_length = 8

cfg = Config.fromfile(config)
labels = [
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

onnx_ckpt = os.path.join(home, 'checkpoints/tsm_1x1x8_sthv2_20220522.onnx')
onnx_model = onnx.load(onnx_ckpt)
onnx_sess = onnxruntime.InferenceSession(onnx_ckpt)

torch_ckpt = os.path.join(
    home,
    'WorkoutDetector/work_dirs/tsm_8_binary_squat_20220607_1956/best_top1_acc_epoch_16.pth'
)
cfg.model.cls_head.num_classes = 2
torch_model = init_recognizer(cfg, torch_ckpt, device=device)


def onnx_inference(inputs: Tensor) -> dict:
    """Inference with ONNX Runtime.
    Args: inputs: Tensor, shape=(1, 8, 3, 224, 224)
    """
    # print('onnx_inference', 'inputs', inputs.shape)
    onnx.checker.check_model(onnx_model)
    # get onnx output
    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer = [node.name for node in onnx_model.graph.initializer]
    net_feed_input = list(set(input_all) - set(input_initializer))
    assert len(net_feed_input) == 1

    onnx_scores = onnx_sess.run(None,
                                {net_feed_input[0]: inputs.cpu().detach().numpy()})[0]
    # print(onnx_scores[0])
    onnx_scores = list(enumerate(onnx_scores[0]))
    onnx_scores.sort(key=lambda x: x[1], reverse=True)
    onnx_text = OrderedDict()
    for i, r in onnx_scores:
        label = labels[i]
        onnx_text[label] = float(r)
    # print(onnx_text)
    return onnx_text


def sample_frames(data, num):
    """Uniformly sample num frames from video, keep order"""
    total = len(data)
    if total <= num:
        # repeat last frame if num > total
        ret = np.vstack(
            [data,
             np.repeat(data[-1], num - total, axis=0).reshape(-1, *data.shape[1:])])
        return ret
    interval = total // num
    indices = np.arange(0, total, interval)[:num]
    for i, x in enumerate(indices):
        rand = np.random.randint(0, interval)
        if i == num - 1:
            upper = total
        else:
            upper = min(interval * (i + 1), total)
        indices[i] = (x + rand) % upper
    assert len(indices) == num, f'len(indices)={len(indices)}'
    ret = data[indices]
    return ret


def torch_inference(cur_data, labels: List[str] = ['down', 'up']) -> dict:
    """Inference with PyTorch.
    Args: cur_data: Tensor, shape=(1, 8, 3, 224, 224)
    """
    cur_data = collate([cur_data], samples_per_gpu=1)
    if next(torch_model.parameters()).is_cuda:
        cur_data = scatter(cur_data, [device])[0]
    with torch.no_grad():
        scores = torch_model(return_loss=False, **cur_data)[0]
    scores = list(enumerate(scores))
    scores.sort(key=lambda x: x[1], reverse=True)
    text = OrderedDict()
    for i, r in scores:
        label = labels[i]
        text[label] = float(r)
    return text


def inference_video(video, single_class=True):
    print('Video:', video)
    capture = cv2.VideoCapture(video)
    if not capture.isOpened():
        print('Could not open video')
        return {'success': False, 'msg': 'Could not open video'}

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
    video_data = dict(img_shape=(height, width),
                      modality='RGB',
                      label=-1,
                      start_index=0,
                      total_frames=len(frames))
    frames = np.array(frames)
    pipeline = Compose(test_pipeline)
    if single_class:
        video_data['imgs'] = sample_frames(frames, sample_length)
        print(video_data['imgs'].shape)  # (8, W, H, 3)
        cur_data = pipeline(video_data)  # (8, 3, 224, 224)
        print(cur_data['imgs'].shape)
        scores = onnx_inference(torch.unsqueeze(cur_data['imgs'], 0))

    else:
        scores = []
        for i in range(len(frames) - sample_length * 2):
            video_data['imgs'] = frames[i:i + 2 * sample_length:2]  # step=2
            cur_data = pipeline(video_data)
            score = torch_inference(cur_data)
            scores += [score] * 2

    return {'success': True, 'data': scores}


def create_video(video, scores: List[OrderedDict]):
    vcap = cv2.VideoCapture(video)
    width = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = vcap.get(cv2.CAP_PROP_FPS)
    tmpfile = os.path.join(home, 'exp', video.split('/')[-1].split('.')[0] + '.webm')
    output_video = cv2.VideoWriter(tmpfile, cv2.VideoWriter_fourcc(*'vp80'), fps,
                                   (int(width), int(height)))

    for i, score in enumerate(scores):
        ret, frame = vcap.read()
        if not ret:
            break
        label = list(score.keys())[0]
        if label == 'down':
            color = (0, 0, 255)
        elif label == 'up':
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)
        cv2.putText(frame, f'{label}: {score[label]}',
                    (int(width * 0.2), int(height * 0.2)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    color, 2)
        output_video.write(frame)
    vcap.release()
    output_video.release()
    return tmpfile


def main(video, mode):
    print('video:', video)
    if mode == 'single':
        single_class = True
    elif mode == 'multi':
        single_class = False
    if not video:
        print('No video input')
        return None
    results = inference_video(video, single_class)
    if not results:
        return None, None
    if not results['success']:
        return None, None
    if single_class:
        return results['data'], None
    output_video = create_video(video, results['data'])
    print('output_video:', output_video)
    return None, output_video


if __name__ == '__main__':

    example_dir = os.path.join(home, 'example_videos')
    example_videos = [os.path.join(example_dir, x) for x in os.listdir(example_dir)]

    demo = gr.Interface(
        fn=main,
        # inputs=[gr.Image(source='webcam', streaming=True,
        #                  type="numpy")],
        inputs=[
            gr.Video(source='upload'),
            gr.Radio(label='Classes in the video',
                     choices=['single', 'multi'],
                     value='single')
        ],
        outputs=["label", "video"],
        examples=[[vid, 'single'] for vid in example_videos],
        title="MMAction2 webcam demo",
        description="Input a video file. Output the recognition result.",
        live=False,
        allow_flagging='never',
    )

    demo.launch()
