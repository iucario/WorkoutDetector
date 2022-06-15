from WorkoutDetector.settings import PROJ_ROOT
import argparse
from bisect import bisect_left
from collections import deque
import os
from typing import Deque, List, Tuple
import PIL
import cv2
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


def inference_image(ort_session: onnxruntime.InferenceSession,
                    frame: np.ndarray,
                    threshold: float = 0.5) -> int:
    frame = data_transform(frame).unsqueeze(0).numpy()
    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: frame}
    ort_outs = ort_session.run(None, ort_inputs)
    score = ort_outs[0][0]
    print(f'{score=}')
    pred = score.argmax()

    return pred


def count_by_image_model(ort_session: onnxruntime.InferenceSession,
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
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'vp80'), fps,
                              (width, height))
    result: Deque[int] = deque(maxlen=7)
    count = 0
    states: List[int] = []
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
<<<<<<< HEAD
        preds: list of size total_frames in the video. If -1, it means no action.

    Returns:
        A tuple of (repetition count, list of frame indices of action end state).

    Note:
        The labels are of pairs. Because that's how I loaded the data.
        E.g. 0 and 1 represent the start and end of the same action.
        We consider action class as well as state changes.
        I can ensemble a standalone action recognition model if things don't work well.

    Algorithm:
        1. If the state changes, and current and previous state are of the same action,
            and are in order, we count the action. For example, if the state changes from
            0 to 1, or 2 to 3, aka even to odd, we count the action.
        
        It means the model have to capture the presice time of state transition.
        Because the model takes 8 continous frames as input.
        Or I doubt it will work well. So multiple time scale should be added.
=======
        preds: list of size total_frames in the video.
    Returns:
        A tuple of (repetition count, list of frame indices of action end state).
>>>>>>> dev
    """

    count = 0
    reps = []
    states: List[int] = []
    for idx, pred in enumerate(preds):
<<<<<<< HEAD
        # if state changed and current and previous state are the same action
        if states and states[-1] != pred:
            if pred % 2 == 1 and states[-1] == pred - 1:
=======
        if states and states[-1] != pred:
            if pred != states[0]:
>>>>>>> dev
                count += 1
                reps.append(idx)
        states.append(pred)
    return count, reps


def write_to_video(video_path: str, output_path: str, preds: List[int]) -> None:
    """Write the predicted count to a video.
    
    Args:
        video_path: path to the video.
        output_path: path to save the output video.
        preds: list of predicted repetition counts.
    """

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if output_path.endswith('.webm'):
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'vp80'), fps,
                              (width, height))
    else:
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                              (width, height))

    count, reps = pred_to_count(preds)
    for idx, res in enumerate(preds):
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


<<<<<<< HEAD
def inference_video(ort_session: onnxruntime.InferenceSession,
                    inputs: np.ndarray,
                    threshold: float = 0.5) -> int:
=======
def inference_video(ort_session: onnxruntime.InferenceSession, inputs: np.ndarray) -> int:
>>>>>>> dev
    """Time shift module inference. 8 frames.

    Args:
        ort_session: ONNX Runtime session. [1, 8, 3, 224, 224]

    Returns:
<<<<<<< HEAD
        int: prediction.
=======
        int: prediction. 0 or 1.
>>>>>>> dev
    """

    inputs = np.stack([data_transform(x) for x in inputs])
    inputs = np.expand_dims(inputs, axis=0)
    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: inputs}
    ort_outs = ort_session.run(None, ort_inputs)
    score = ort_outs[0][0]
    pred = score.argmax()
<<<<<<< HEAD
    print(score, pred)
=======
>>>>>>> dev
    return pred


def count_by_video_model(ort_session: onnxruntime.InferenceSession,
                         video_path: str,
                         ground_truth: list,
                         output_path: str = None) -> Tuple[int, int]:
    """Evaluate repetition count on a video, using video classification model.
    
    Args:
        ort_session: ONNX Runtime session. [1, 8, 3, 224, 224]
        video_path: path to the video.
        ground_truth: list of ground truth repetition counts.
        output_path: path to save the output video.
    
    Returns:
        Tuple[int, int]: count and ground truth count.
    """

    video_name = os.path.basename(video_path)
    print(f'{video_name}')
    cap = cv2.VideoCapture(video_path)
    input_queue: Deque[int] = deque(maxlen=8)
    result = []
    count = 0
    states: List[int] = []
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
            pred = inference_video(ort_session, input_clip)
            result += [pred] * 8
            if states and states[-1] != pred:
                if pred != states[0]:
                    count += 1
                    reps.append(frame_idx)
            states.append(pred)
            input_queue.clear()
    gt_count = len(ground_truth) // 2
    error = abs(count - gt_count)
    print(f'count={count}, gt_count={gt_count}',
          f'error={error} error_rate={error/max(1, gt_count):.2}')
    cap.release()
    if output_path is not None:
        write_to_video(video_path, output_path, result)
    return count, gt_count


def infer_dataset(ort_session: onnxruntime.InferenceSession, action_name: str,
                  model_type: str) -> None:
    data_root = os.path.join(PROJ_ROOT, 'data')
    dataset = RepcountDataset(root=data_root, split='test')
    action_df = dataset.df[dataset.df['class_'] == action_name]
    names = action_df['name'].values

    total_count = 0
    total_gt_count = 0
    for i in range(len(names)):
        rand_video = os.path.join(data_root, 'RepCount/videos/test', names[i])

        if action_df['count'].values[i]:
            gt = list(map(int, action_df['reps'].values[i].split(' ')))
        else:
            gt = []
        if model_type == 'image':
            count, gt_count = count_by_image_model(ort_session,
                                                   rand_video,
                                                   gt,
                                                   output_path=None)
        elif model_type == 'video':
            count, gt_count = count_by_video_model(ort_session=ort_session,
                                                   video_path=rand_video,
                                                   ground_truth=gt,
                                                   output_path=None)
        total_count += count
        total_gt_count += gt_count


def main(args) -> None:
<<<<<<< HEAD
=======
    action_name = args.action
>>>>>>> dev
    onnx_path = args.onnx
    ort_session = onnxruntime.InferenceSession(
        onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    if args.video:
        video_path = args.video
        if args.model_type == 'image':
            count_by_image_model(ort_session,
                                 video_path,
                                 ground_truth=[],
                                 output_path=args.output)
        elif args.model_type == 'video':
            count_by_video_model(ort_session,
                                 video_path,
                                 ground_truth=[],
                                 output_path=args.output)
    else:
<<<<<<< HEAD
        action_name = args.action
=======
>>>>>>> dev
        infer_dataset(ort_session, action_name, model_type=args.model_type)


def mmlab_infer(args):
    cfg_path = os.path.join(PROJ_ROOT, '/WorkoutDetector/tsm_config.py')
    model = init_recognizer(cfg_path, args.checkpoint, device='cuda')
    results = inference_recognizer(model, args.video)
    # TODO


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate RepCount')
<<<<<<< HEAD
    parser.add_argument('--onnx', help='onnx path')
    parser.add_argument('-i', '--video', help='video path', required=False)
    parser.add_argument('-t', '--threshold', help='threshold', type=float, default=0.5)
    parser.add_argument('-ckpt', '--checkpoint', help='checkpoint path', required=False)
=======
    parser.add_argument('-ckpt', '--checkpoint', help='checkpoint path', required=False)
    parser.add_argument('--onnx', help='onnx path')
    parser.add_argument('--video', help='video path', required=False)
>>>>>>> dev
    parser.add_argument('-o', '--output', help='output path', required=False)
    parser.add_argument('-m',
                        '--model-type',
                        help='evaluate using image/video model',
                        default='video',
                        choices=['image', 'video'])
<<<<<<< HEAD
    parser.add_argument('-a',
                        '--action',
                        help='action name',
                        default='situp',
                        choices=[
                            'situp', 'push_up', 'pull_up', 'jump_jack', 'squat',
                            'front_raise', 'all'
                        ])
    args = parser.parse_args([
        '--onnx',
        'checkpoints/tsm_video_all.onnx',
        '--threshold',
        '0.5',
        '--video',
        'data/RepCount/videos/test/situp_1.mp4',
    ])
=======
    parser.add_argument(
        '-a',
        '--action',
        help='action name',
        default='situp',
        choices=['situp', 'push_up', 'pull_up', 'jump_jack', 'squat', 'front_raise'])
    args = parser.parse_args()
>>>>>>> dev

    main(args)
    # mmlab_infer(args)