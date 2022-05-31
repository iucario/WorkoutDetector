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

import gradio as gr
import warnings
import yaml

user_cfg = yaml.safe_load(
    open(
        os.path.join(
            os.path.dirname(__file__),
            'utils/config.yml')))

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
home = user_cfg['proj_root']
print('HOME', home)
config = os.path.join(
    home, 'mmaction2/configs/recognition/tsm/tsm_my_config.py')
sample_length = 8

cfg = Config.fromfile(config)
labels = [
    'front_raise', 'pull_up', 'squat', 'bench_pressing', 'jumping_jack',
    'situp', 'push_up', 'battle_rope', 'exercising_arm', 'lunge',
    'mountain_climber', ]


onnx_ckpt = os.path.join(home, 'checkpoints/tsm_1x1x8_sthv2_20220522.onnx')
onnx_model = onnx.load(onnx_ckpt)
onnx_sess = onnxruntime.InferenceSession(onnx_ckpt)


def onnx_inference(inputs: Tensor) -> List[float]:
    """Inference with ONNX Runtime.
    Args: inputs: Tensor, shape=(1, 16, 3, 224, 224)
    """
    # print('onnx_inference', 'inputs', inputs.shape)
    onnx.checker.check_model(onnx_model)
    # get onnx output
    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer = [
        node.name for node in onnx_model.graph.initializer
    ]
    net_feed_input = list(set(input_all) - set(input_initializer))
    assert len(net_feed_input) == 1

    onnx_scores = onnx_sess.run(
        None, {net_feed_input[0]: inputs.cpu().detach().numpy()})[0]
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
    video_data = dict(img_shape=(height, width), modality='RGB',
                      label=-1, start_index=0, total_frames=len(frames))
    frames = np.array(frames)
    video_data['imgs'] = sample_frames(frames, sample_length)
    print(video_data['imgs'].shape)  # (16, W, H, 3)
    pipeline = Compose(test_pipeline)
    cur_data = pipeline(video_data)  # (16, 3, 224, 224)
    print(cur_data['imgs'].shape)
    onnx_scores = onnx_inference(torch.unsqueeze(cur_data['imgs'], 0))

    return {'success': True, 'data': onnx_scores}


def main(inputs):
    print('Inputs:', inputs)
    if not inputs:
        print('No video input')
        return None
    video = inputs
    results = inference_video(video)
    if not results:
        return None
    if not results['success']:
        return None
    return results['data']


if __name__ == '__main__':

    # demo = gr.Blocks()
    # with demo:
    #     with gr.Row():
    #         inp = gr.Webcam(source='webcam', streaming=True, type="numpy")
    #         out = gr.Label(num_top_classes=5)
    #     btn = gr.Button("Run")
    #     btn.click(fn=main, inputs=inp, outputs=out)

    demo = gr.Interface(
        fn=main,
        # inputs=[gr.Image(source='webcam', streaming=True,
        #                  type="numpy")],
        inputs=[gr.Video(source='upload')],
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
