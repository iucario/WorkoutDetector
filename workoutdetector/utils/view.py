import os
from os.path import join as osj
from typing import List
from torch import Tensor
import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.io import read_video
from workoutdetector.datasets import RepcountHelper
from workoutdetector.settings import PROJ_ROOT, REPCOUNT_ANNO_PATH

helper = RepcountHelper(osj(PROJ_ROOT, 'data/RepCount'), REPCOUNT_ANNO_PATH)
DATA = helper.get_rep_data(split=['train', 'val', 'test'],
                           action=['push_up', 'front_raise'])


def show_one_video(video_name: str) -> List[Image.Image]:
    """Read video and anno, then show frames of interest.
    
    Args:
        video_path: path to video
    Returns:
        list of PIL images, annotated repetitinon start and end frames, and middle.
    """

    item = DATA[video_name]
    path = item.video_path
    reps = item.reps
    count = item.count
    video = read_video(path)
    frames = video[0]
    meta = video[2]
    inds = []
    for s, e in zip(reps[::2], reps[1::2]):
        mid = (s + e) // 2
        inds += [s, mid, e]
    return [frames[i].numpy() for i in inds]


def load_examples() -> List[list]:
    """Load annotations"""
    ret = []
    for i, item in enumerate(DATA.values()):
        ret.append([i, item.video_name, item.split, item.class_])
    return ret


def classify(video_path: str) -> dict:
    """Video classification function"""

    return {'push_up': 0, 'front_raise': 1}


def display(idx: int, name: str, split: str, action: str) -> List[Image.Image]:
    """Gradio main function"""
    return show_one_video(name)


if __name__ == '__main__':
    demo = gr.Interface(
        fn=classify,
        inputs=[gr.Video(source='upload')],
        outputs=["label"],
        live=False,
    )


def func1():
    examples = load_examples()

    demo = gr.Interface(fn=display,
                        inputs=[
                            gr.Number(label="id"),
                            gr.Text(label="video_name"),
                            gr.Radio(label='split',
                                     value='train',
                                     choices=['train', 'val', 'test']),
                            gr.Radio(label="action",
                                     value='push_up',
                                     choices=['push_up', 'front_raise'])
                        ],
                        outputs=gr.Gallery(label="Start, mid and end frames"),
                        examples=examples)

    demo.launch()
