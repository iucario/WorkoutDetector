import argparse
from bisect import bisect_left
from collections import deque
import os
from typing import List, Tuple
import PIL
import cv2
from cv2 import transform
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from WorkoutDetector.datasets import RepcountDataset
import onnx
import onnxruntime

from mmaction.apis import inference_recognizer, init_recognizer

onnxruntime.set_default_logger_severity(3)

data_transform = T.Compose([
    T.ToPILImage(),
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

COLORS = {
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0),
    'yellow': (0, 255, 255),
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'orange': (0, 165, 255),
    'purple': (255, 0, 255),
    'pink': (255, 192, 203),
    'brown': (165, 42, 42),
    'cyan': (0, 255, 255),
    'magenta': (255, 0, 255),
    'lime': (0, 255, 0),
}


def inference_image(ort_session, frame) -> int:
    # frame = cv2.resize(frame, (224, 224))
    # frame = np.transpose(frame, (2, 0, 1))
    # frame = np.expand_dims(frame, axis=0)
    # frame = frame.astype(np.float32)
    frame = data_transform(frame).unsqueeze(0).numpy()
    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: frame}
    ort_outs = ort_session.run(None, ort_inputs)
    score = ort_outs[0][0]
    pred = score.argmax()

    return pred


def inference_video(ort_session, video_path, output_path) -> None:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                          (width, height))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pred = inference_image(ort_session, frame)
        text = str(pred)
        if pred == 1:
            color = (0, 0, 255)
        elif pred == 0:
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)
        cv2.putText(frame, text, (int(width * 0.3), int(height * 0.3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        out.write(frame)
    cap.release()
    out.release()


def evaluate_by_image(ort_session: onnxruntime.InferenceSession,
                      video_path: str,
                      ground_truth: list,
                      output_path: str = None) -> Tuple[int, int]:
    """Evaluate repetition count on a video, using image classification model.
    
    Args:
        ort_session: ONNX Runtime session.
        video_path: video to be evaluated.
        ground_truth: list, column `reps` in `annotation.csv`.
        output_path: path to save the output video. If None, no video will be saved.

    Returns:
        Tuple[int, int]: (repetition count, number of frames of action).

    Note:
        Voting is used to determine the repetition count.
    """

    print(f'{video_path}')
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if output_path:
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                              (width, height))
    result = deque(maxlen=7)
    count = 0
    states = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        frame_idx += 1
        if not ret:
            break
        curr_pred = inference_image(ort_session, frame)
        result.append(curr_pred)
        pred = sum(result) > len(result) // 2  # vote of frames
        if not states:
            states.append(pred)
        elif states[-1] != pred:
            states.append(pred)
            if pred != states[0]:
                count += 1
        if output_path:
            text = str(curr_pred)
            if pred == 1:
                color = COLORS['red']
            else:
                color = COLORS['green']
            cv2.putText(frame, text, (int(width * 0.2), int(height * 0.2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f'pred {count}', (int(width * 0.2), int(height * 0.4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (52, 235, 177), 2)
            gt_count = bisect_left(ground_truth[1::2], frame_idx)
            cv2.putText(frame, f'true {gt_count}', (int(width * 0.2), int(height * 0.6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (180, 235, 52), 2)
            out.write(frame)
    error = abs(count - len(ground_truth) // 2)
    gt_count = len(ground_truth) // 2

    print(f'{error=} {count=} {gt_count=}')

    cap.release()
    if output_path:
        out.release()
    return count, gt_count


def pred_to_count(preds: List[int]) -> Tuple[int, List[int]]:
    """Convert a list of predictions to a repetition count.
    
    Args:
        preds: list of size total_frames in the video.
    Returns:
        A tuple of (repetition count, list of frame indices of action end state).
    """

    count = 0
    reps = []
    states = []
    for idx, pred in enumerate(preds):
        if states and states[-1] != pred:
            if pred != states[0]:
                count += 1
                reps.append(idx)
        states.append(pred)
    return count, reps


def write_to_video(video_path, output_path, result):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                          (width, height))

    count, reps = pred_to_count(result)
    for idx, res in enumerate(result):
        ret, frame = cap.read()
        if not ret:
            break
        count_idx = bisect_left(reps, idx)
        cv2.putText(frame, str(res), (int(width * 0.2), int(height * 0.2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS['cyan'], 2)
        cv2.putText(frame, f'count {count_idx}', (int(width * 0.2), int(height * 0.4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS['pink'], 2)
        out.write(frame)
    cap.release()
    out.release()


def onnx_inference(ort_session, inputs: torch.Tensor) -> int:
    """tsm 8 frames inference.

    Args:
        ort_session: ONNX Runtime session. [1, 8, 3, 224, 224]

    Returns:
        int: prediction. 0 or 1.
    """

    inputs = np.stack([data_transform(x) for x in inputs])
    inputs = np.expand_dims(inputs, axis=0)
    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: inputs}
    ort_outs = ort_session.run(None, ort_inputs)
    score = ort_outs[0][0]
    pred = score.argmax()
    return pred


def evaluate(ort_session,
             video_path,
             ground_truth: list,
             output_dir=None) -> Tuple[int, int]:
    """Evaluate repetition count on a video, using video classification model.
    
    Args:
        ort_session: ONNX Runtime session. [1, 8, 3, 224, 224]
    
    Returns:
        Tuple[int, int]: count and ground truth count.
    """
    video_name = os.path.basename(video_path)
    output_path = os.path.join(output_dir, video_name)
    cap = cv2.VideoCapture(video_path)
    input_queue = deque(maxlen=8)
    result = []
    count = 0
    states = []
    reps = []  # frame indices of action end state, start from 1
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        frame_idx += 1
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_queue.append(frame)
        if len(input_queue) == 8:
            input_clip = np.array(input_queue)
            pred = onnx_inference(ort_session, input_clip)
            result += [pred] * 8
            if states and states[-1] != pred:
                if pred != states[0]:
                    count += 1
                    reps.append(frame_idx)
            states.append(pred)
            input_queue.clear()
    gt_count = len(ground_truth) // 2
    error = abs(count - gt_count)
    print(
        f'count={count}, gt_count={gt_count} '\
            f'error={error} error rate={error/max(1, gt_count):.2}'
    )
    cap.release()
    if output_path is not None:
        write_to_video(video_path, output_path, result)
        with open(output_path + '.txt', 'w') as f:
            f.write('result ' + ' '.join(map(str, result)) + '\n')
            f.write('reps ' + ' '.join(map(str, reps)) + '\n')
            f.write(f'count={count} gt_count={gt_count}\n')
            f.write(f'error={error} error rate={error/max(1, gt_count):.2}\n')
    return count, gt_count


def main(args):
    providers = [
        ('CUDAExecutionProvider', {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'gpu_mem_limit': 4 * 1024 * 1024 * 1024,
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
        }),
        'CPUExecutionProvider',
    ]
    action_name = args.action
    if args.image:
        onnx_path = os.path.join('/home/umi/projects/WorkoutDetector/checkpoints/',
                                 f'{action_name}_20220610.onnx')
    data_root = '/home/umi/projects/WorkoutDetector/data'
    dataset = RepcountDataset(root=data_root, split='test')
    action = dataset.df[dataset.df['class_'] == action_name]
    l = action['name'].values

    ort_session = onnxruntime.InferenceSession(onnx_path, providers=providers)

    total_count = 0
    total_gt_count = 0
    for i in range(len(l)):
        rand_video = os.path.join(data_root, 'RepCount/videos/test', l[i])

        gt = list(map(
            int,
            action['reps'].values[i].split(' '))) if action['count'].values[i] else []
        if args.image:
            count, gt_count = evaluate_by_image(ort_session,
                                                rand_video,
                                                gt,
                                                output_path=None)
        else:
            count, gt_count = evaluate(
                ort_session=ort_session,
                video_path=rand_video,
                ground_truth=gt,
                output_dir=f'/mnt/d/infer/video_cls_{action_name}/')
        total_count += count
        total_gt_count += gt_count


def mmlab_infer(args):
    cfg_path = '/home/umi/projects/WorkoutDetector/WorkoutDetector/tsm_config.py'
    model = init_recognizer(cfg_path, args.checkpoint, device='cuda')
    results = inference_recognizer(model, args.video)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate RepCount')
    parser.add_argument('-ckpt', '--checkpoint', help='checkpoint path', required=False)
    parser.add_argument('--video', help='video path', required=False)
    parser.add_argument('-im',
                        '--image-model',
                        dest='image',
                        action='store_true',
                        help='evaluate using image model')
    parser.add_argument(
        '-a',
        '--action',
        help='action name',
        default='situp',
        choices=['situp', 'push_up', 'pull_up', 'jump_jack', 'squat', 'front_raise'])
    args = parser.parse_args()

    main(args)
    # mmlab_infer(args)