import argparse
import os
from bisect import bisect_left
from collections import deque
from typing import Deque, List, Optional, Tuple, Union

import cv2
from mmaction.apis.inference import inference_recognizer
import numpy as np
import onnx
import onnxruntime
import pandas as pd
import PIL
import torch
import torchvision.transforms as T
from mmaction.apis import init_recognizer
from mmaction.datasets.pipelines import Compose
from mmcv import Config, DictAction
from mmcv.parallel import collate, scatter
from WorkoutDetector.datasets import RepcountHelper
from WorkoutDetector.settings import PROJ_ROOT, REPCOUNT_ANNO_PATH

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


def inference_image(model: Union[onnxruntime.InferenceSession, torch.nn.Module],
                    frame: np.ndarray,
                    threshold: float = 0.5) -> int:
    assert type(model) is onnxruntime.InferenceSession
    frame = data_transform(frame).unsqueeze(0).numpy()  # type: ignore
    input_name = model.get_inputs()[0].name
    ort_inputs = {input_name: frame}
    ort_outs = model.run(None, ort_inputs)
    score = ort_outs[0][0]
    print(f'{score=}')
    pred = score.argmax()

    return pred


def count_by_image_model(model: Union[onnxruntime.InferenceSession, torch.nn.Module],
                         video_path: str,
                         ground_truth: Optional[List[int]] = None,
                         output_path: Optional[str] = None) -> Tuple[int, List[int]]:
    """Evaluate repetition count on a video, using image classification model.
    
    Args:
        ort_session: ONNX Runtime session.
        video_path: video to be evaluated.
        ground_truth: list, column `reps` in `annotation.csv`.
        output_path: path to save the output video. If None, no video will be saved.

    Returns:
        Tuple[int, List[int]]: (repetition count, predicted reps).

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
        if not ret:
            break
        curr_pred = inference_image(model, frame)
        result.append(curr_pred)
        pred = sum(result) > len(result) // 2  # vote of frames
        states.append(pred)
        frame_idx += 1
    cap.release()

    count, reps = pred_to_count(step=8, preds=states)
    gt_count = len(ground_truth) // 2 if ground_truth else -1
    correct = (abs(count - gt_count) <= 1)
    print(f'count={count} gt_count={gt_count} correct={correct}')

    if output_path and out.isOpened():  # type: ignore
        out.release()  # type: ignore
    return count, reps


def pred_to_count(step: int, preds: List[int]) -> Tuple[int, List[int]]:
    """Convert a list of predictions to a repetition count.
    
    Args:
        step: step size of the predictions.
        preds: list of size total_frames//step in the video. If -1, it means no action.

    Returns:
        A tuple of (repetition count, 
        list of preds of action start and end states, e.g. start_1, end_1, start_2, end_2, ...)

    Note:
        The labels are of pairs. Because that's how I loaded the data.
        E.g. 0 and 1 represent the start and end of the same action.
        We consider action class as well as state changes.
        I can ensemble a standalone action recognition model if things don't work well.

    Algorithm:
        1. If the state changes, and current and previous state are of the same action,
            and are in order, we count the action. For example, if the state changes from
            0 to 1, or 2 to 3, aka even to odd, we count the action.
        
        It means the model has to capture the presice time of state transition.
        Because the model takes 8 continous frames as input.
        Or I doubt it will work well. So multiple time scale should be added.
    """

    count = 0
    reps = []  # start_1, end_1, start_2, end_2, ...
    states: List[int] = []
    prev_state_start_idx = 0
    for idx, pred in enumerate(preds):
        # if state changed and current and previous state are the same action
        if pred > -1 and states and states[-1] != pred:
            if pred % 2 == 1 and states[-1] == pred - 1:
                count += 1
                reps.append(prev_state_start_idx * step)
                reps.append(idx * step)
        states.append(pred)
        prev_state = states[prev_state_start_idx]
        if pred != prev_state:  # new state, new start index
            prev_state_start_idx = idx

    assert count * 2 == len(reps)
    return count, reps  # len(rep) * step <= len(frames), last not full queue is discarded


def write_to_video(video_path: str,
                   output_path: str,
                   reps: List[int],
                   states: List[int],
                   step: int = 8) -> None:
    """Write the predicted count to a video.
    
    Args:
        video_path: path to the video.
        output_path: path to save the output video.
        reps: list of predicted start and end indices.
        states: list of predicted states.
        step: step size of the predictions.
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

    for idx, res in enumerate(np.repeat(states, step)):
        ret, frame = cap.read()
        if not ret:
            break
        count_idx = bisect_left(reps[::2], idx)
        cv2.putText(frame, str(res), (int(width * 0.2), int(height * 0.2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS['cyan'], 2)
        cv2.putText(frame, f'count {count_idx}', (int(width * 0.2), int(height * 0.4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS['red'], 2)
        out.write(frame)
    cap.release()
    out.release()


def inference_video(model: Union[onnxruntime.InferenceSession, torch.nn.Module],
                    inputs: np.ndarray,
                    threshold: float = 0.5) -> int:
    """Time shift module inference. 8 frames.

    Args:
        model: ONNX Runtime session or PyTorch.
        inputs: np.ndarray of shape [1, 8, 3, 224, 224]
        is_torch: if True, use PyTorch. Else, use ONNX Runtime.
        threshold: threshold for bbox. # TODO: implement this.

    Returns:
        int: prediction.
    """
    if type(model) is onnxruntime.InferenceSession:
        inputs = np.stack([data_transform(x) for x in inputs])
        inputs = np.expand_dims(inputs, axis=0)
        input_name = model.get_inputs()[0].name
        ort_inputs = {input_name: inputs}
        ort_outs = model.run(None, ort_inputs)
        score = ort_outs[0][0]
        pred = score.argmax()
    else:  # use mmlab inference
        input_clip = np.array(inputs)
        score = inference_recognizer(model, input_clip)
        pred = score[0][0]
    # print('score', list(score))
    return pred


def count_by_video_model(model: Union[onnxruntime.InferenceSession, torch.nn.Module],
                         video_path: str,
                         ground_truth: Optional[list] = None,
                         output_path: Optional[str] = None) -> Tuple[int, List[int]]:
    """Evaluate repetition count on a video, using video classification model.
    
    Args:
        ort_session: ONNX Runtime session. [1, 8, 3, 224, 224]
        video_path: path to the video.
        ground_truth: list of ground truth repetition counts.
        output_path: path to save the output video.
    
    Returns:
        Tuple[int, List[int]]: predicted count and reps.
    
    Note:
        The current implementation is not online inference. Because it's in debug mode.
        Will be updated when the accuracy is good enough.
    """

    video_name = os.path.basename(video_path)
    print(f'{video_path}')
    cap = cv2.VideoCapture(video_path)
    input_queue: Deque[np.ndarray] = deque(maxlen=8)
    count = 0
    states: List[int] = []  # onnx preds
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_queue.append(frame)
        if len(input_queue) == 8:
            input_clip = np.array(input_queue)
            pred = inference_video(model, input_clip)
            states.append(pred)
            input_queue.clear()
        frame_idx += 1
    cap.release()

    count, reps = pred_to_count(step=8, preds=states)
    gt_count = len(ground_truth) // 2 if ground_truth else -1
    correct = (abs(gt_count - count) <= 1)
    print(f'count={count}, gt_count={gt_count}, correct={correct}')
    if output_path is not None:
        write_to_video(video_path, output_path, reps, states=states, step=8)
    return count, reps


def infer_dataset(model: Union[onnxruntime.InferenceSession, torch.nn.Module],
                  action: List[str],
                  model_type: str = 'video',
                  output_dir: Optional[str] = None) -> None:
    """Inference on a dataset test split.
    
    Args:
        model: ONNX Runtime session or PyTorch model. [1, 8, 3, 224, 224]
        action: list of action name.
        model_type: model type. Image or video model.
        output_dir: path to save the output videos and result csv.
    """

    data_root = os.path.join(PROJ_ROOT, 'data/RepCount/')
    assert data_root is not None
    helper = RepcountHelper(data_root, REPCOUNT_ANNO_PATH)
    repcount_items = helper.get_rep_data(split=['test'], action=action)
    SPLIT = 'test'
    pred_dict = dict()
    for name, item in repcount_items.items():
        assert os.path.exists(item['video_path']), f'{item["video_path"]} not exists'
        if output_dir is not None:
            assert os.path.isdir(output_dir)
            assert name.endswith('.mp4')
            output_path = os.path.join(output_dir, name)
        else:
            output_path = None
        if model_type == 'video':
            count, reps = count_by_video_model(model,
                                               item.video_path,
                                               ground_truth=item.reps,
                                               output_path=output_path)
        elif model_type == 'image':
            count, reps = count_by_image_model(model,
                                               item.video_path,
                                               ground_truth=item.reps,
                                               output_path=output_path)
        else:
            raise ValueError(f'Invalid model type: {model_type}')
        pred_dict[name] = count  # Only implemented count evaluation for now.
    mae, obo_acc, eval_res = helper.eval_count(pred_dict, action=action, split=[SPLIT])
    print(f'MAE={mae}, OBO_ACC={obo_acc}, SPLIT=test, ACTION={action}')
    if output_dir is not None:  # write to csv
        res = []
        for item in eval_res.values():
            dict_ = item.__dict__
            dict_.pop('video_path')
            dict_.pop('frames_path')
            res.append(dict_)
        df = pd.DataFrame.from_dict(res,)
        df.to_csv(os.path.join(output_dir, f'eval_count_{model_type}_model.csv'))


def main(args) -> None:
    if args.mmlab:
        cfg_path = os.path.join(
            PROJ_ROOT, 'WorkoutDetector/configs/tsm_MultiActionRepCount_sthv2.py')
        model = init_recognizer(cfg_path, args.checkpoint, device='cuda')
    else:
        assert args.onnx is not None
        onnx_path = args.onnx
        model = onnxruntime.InferenceSession(
            onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    if not args.eval and args.video is not None:
        video_path = args.video
        if args.model_type == 'image':
            count_by_image_model(model,
                                 video_path,
                                 ground_truth=[],
                                 output_path=args.output)
        elif args.model_type == 'video':
            count_by_video_model(model,
                                 video_path,
                                 ground_truth=[],
                                 output_path=args.output)
    elif args.eval:
        CLASSES = ['situp', 'push_up', 'pull_up', 'jump_jack', 'squat', 'front_raise']
        if args.action == 'all':
            action = CLASSES
        else:
            action = [args.action]
        infer_dataset(model,
                      action=action,
                      model_type=args.model_type,
                      output_dir=args.output)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate RepCount')
    parser.add_argument('--onnx', help='onnx path', required=False)
    parser.add_argument('--mmlab', help='use mmlab model', action='store_true')
    parser.add_argument('-i', '--video', help='video path', required=False)
    parser.add_argument('--eval', help='evaluate dataset', action='store_true')
    parser.add_argument('-t', '--threshold', help='threshold', type=float, default=0.5)
    parser.add_argument('-ckpt', '--checkpoint', help='checkpoint path', required=False)
    parser.add_argument('-o',
                        '--output',
                        help='output path. If evaluate dataset, it is output_dir',
                        required=False)
    parser.add_argument('-m',
                        '--model-type',
                        help='evaluate using image/video model',
                        default='video',
                        choices=['image', 'video'])
    parser.add_argument('-a',
                        '--action',
                        help='action name',
                        default='situp',
                        choices=[
                            'situp', 'push_up', 'pull_up', 'jump_jack', 'squat',
                            'front_raise', 'all'
                        ])

    args = parser.parse_args(argv)
    return args


if __name__ == '__main__':
    example_args = [
        '--onnx', 'checkpoints/tsm_video_all_20220616.onnx', '--threshold', '0.5',
        '--video', 'data/RepCount/videos/test/stu1_27.mp4'
    ]
    example_dataset = [
        '--onnx', 'checkpoints/tsm_video_all_20220616.onnx', '--eval', '--output', 'exp/',
        '--model-type', 'video', '--action', 'all'
    ]
    example_mmlab = [
        '--mmlab', '-ckpt', 'checkpoints/tsm_video_all.pth', '--eval', '--output', 'exp/',
        '--model-type', 'video', '--action', 'pull_up'
    ]
    args = parse_args()
    main(args)
