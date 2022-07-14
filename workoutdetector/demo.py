import math
import os
import subprocess
import sys
import tempfile
import warnings
from collections import OrderedDict
from typing import Any, Dict, List, Tuple

import cv2
import gradio as gr
import numpy as np
import onnx
import onnxruntime
import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.io import read_video

from workoutdetector.datasets import RepcountHelper, build_test_transform
from workoutdetector.settings import PROJ_ROOT, REPCOUNT_ANNO_PATH
from workoutdetector.utils import count_by_image_model, count_by_video_model

helper = RepcountHelper(os.path.join(PROJ_ROOT, 'data/RepCount'), REPCOUNT_ANNO_PATH)
DATA = helper.get_rep_data(split=['val', 'test'], action=['all'])

warnings.filterwarnings('ignore')
onnxruntime.set_default_logger_severity(3)

action_11_labels = [
    'front_raise', 'pull_up', 'squat', 'bench_pressing', 'jumping_jack', 'situp',
    'push_up', 'battle_rope', 'exercising_arm', 'lunge', 'mountain_climber'
]
rep_12_labels = [
    'situp 1', 'situp 2', 'push_up 1', 'push_up 2', 'pull_up 1', 'pull_up 2',
    'jump_jack 1', 'jump_jack 2', 'squat 1', 'squat 2', 'front_raise 1', 'front_raise 2'
]

rep_6_labels = ['situp', 'push_up', 'pull_up', 'jump_jack', 'squat', 'front_raise']

action_11_ckpt = 'checkpoints/action-11/tsm_1x1x8_sthv2_20220522.onnx'
rep_12_ckpt = 'checkpoints/repcount-12/rep_12_20220705_220720.onnx'
ort_session = onnxruntime.InferenceSession(rep_12_ckpt,
                                           providers=['CUDAExecutionProvider'])
transform = build_test_transform(person_crop=False)


def onnx_inference(x: Tensor, labels: List[str] = rep_12_labels) -> Dict[str, float]:
    """Inference with ONNX Runtime. Output format is for gr.Label
    
    Args: 
        x: Tensor, shape=(1, 8, 3, 224, 224)
        labels: List of labels, default is `rep_12_labels`

    Returns:
        Dict of (label, score), ordered by score, descending.
    """
    # print('onnx_inference', 'inputs', inputs.shape)
    x = x.unsqueeze(0)
    assert x.shape == (1, 8, 3, 224, 224)
    input_name = ort_session.get_inputs()[0].name
    out = ort_session.run(None, {input_name: x.numpy()})
    softmax = F.softmax(torch.tensor(out[0]), dim=1)
    onnx_scores = list(enumerate(softmax.numpy()[0]))
    onnx_scores.sort(key=lambda x: x[1], reverse=True)
    print(onnx_scores)
    text = dict()
    for i, r in onnx_scores:
        label = labels[i]
        text[label] = float(r)
    # print(onnx_text)
    return text


def sample_frames(data: np.ndarray, num: int = 8) -> np.ndarray:
    """Uniformly sample num frames from video data."""

    total = len(data)
    if total < num:
        # repeat frames if total < num
        repeats = math.ceil(num / total)
        new_inds = [x for x in range(total) for _ in range(repeats)]
        total = len(new_inds)
    interval = total // num
    indices = np.arange(0, total, interval)[:num]
    assert len(indices) == num, f'len(indices)={len(indices)}'
    ret = data[indices]
    return ret


def inference_video_reps(video: str, model_type: str = 'image') -> Tuple[int, str]:
    """Counting repetitions from one video.
    
    Args:
        video: str, video path
        model_type: str, 'image' or 'video'
    
    Returns:
        Tuple of (int, TemporaryFile), (repetitions, output_file)
    """

    with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as tmp_file:
        if model_type == 'image':
            count, _ = count_by_image_model(onnx_image_count_sess,
                                            video,
                                            ground_truth=[],
                                            output_path=tmp_file.name)
        elif model_type == 'video':
            count, _ = count_by_video_model(onnx_video_count_sess,
                                            video,
                                            ground_truth=[],
                                            output_path=tmp_file.name)
        return count, tmp_file.name


def create_video(video, scores: List[OrderedDict]) -> str:
    cap = cv2.VideoCapture(video)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    tmpfile = os.path.join(PROJ_ROOT, 'exp', video.split('/')[-1].split('.')[0] + '.webm')
    output_video = cv2.VideoWriter(tmpfile, cv2.VideoWriter_fourcc(*'vp80'), fps,
                                   (int(width), int(height)))

    for i, score in enumerate(scores):
        ret, frame = cap.read()
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
    cap.release()
    output_video.release()
    return tmpfile


def main(video: str, model_type: str, name: str) -> gr.Label:
    print('video:', video, model_type, name)
    vid = read_video(video)[0]
    x = transform(vid.permute(0, 3, 1, 2))
    x = sample_frames(x, num=8)
    d = onnx_inference(x, labels=rep_12_labels)
    return d


def load_examples() -> List[list]:
    """Load annotations"""
    ret = []
    for i, item in enumerate(DATA.values()):
        ret.append([i, item.video_name, item.class_])
    return ret


if __name__ == '__main__':

    example_dir = 'data/RepCount/rep_video/test'
    example_videos = [os.path.join(example_dir, x) for x in os.listdir(example_dir)]

    demo = gr.Interface(
        fn=main,
        inputs=[
            gr.Video(source='upload'),
            gr.Radio(label='Model',
                     choices=['repcount 12 classes', 'action recognition 6 classes'],
                     value='repcount 12 classes'),
            gr.Text(),
        ],
        outputs=["label"],
        examples=[
            [v, 'repcount 12 classes', os.path.basename(v)] for v in example_videos
        ],
        title="WorkoutDetector demo",
        live=False,
    )

    demo.launch()
