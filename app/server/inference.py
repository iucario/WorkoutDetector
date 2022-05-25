
import warnings
import cv2
import numpy as np
import torch
from mmaction.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter


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
