import argparse
import json
import os
import os.path as osp
import time
from bisect import bisect_left
from collections import deque
from os.path import join as osj
from typing import Callable, Deque, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import onnx
import onnxruntime
import pandas as pd
import torch
import torchvision.transforms as T
from torch import Tensor, nn
from torchvision.io import read_video
from workoutdetector.datasets import (Pipeline, RepcountHelper, build_test_transform)
from workoutdetector.settings import PROJ_ROOT, REPCOUNT_ANNO_PATH

onnxruntime.set_default_logger_severity(3)

data_transform = T.Compose([
    T.ToPILImage(),
    # T.ConvertImageDtype(dtype=torch.float32),
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
    'orange': (12, 136, 237),
}


def save_scores_to_json(scores: List[np.ndarray], output_path: str, video_path: str,
                        step: int) -> None:
    """Save the prediction scores to a json file.
    
    Args:
        scores (List[numpy.ndarray]): prediction scores.
        output_path (str): path to save the json file.
        Will apppend '.json' if not ends with '.json'.
        video_path (str): path to the input video.
        step (int): step size of the predictions.
    """

    if not output_path.endswith('.json'):
        output_path += '.json'
    assert not osp.exists(output_path), f'{output_path} already exists.'
    d = {'video_path': video_path, 'step': step}
    s = {}
    for i, score in enumerate(scores):
        s[i] = score.tolist()
    d['scores'] = s
    json.dump(d, open(output_path, 'w'))


def write_to_video(video_path: str,
                   output_path: str,
                   reps: List[int],
                   states: List[int],
                   step: int = 8) -> None:
    """Write the predicted count to a video. '.mp4' will be added if no extension is given.
    
    Args:
        video_path: path to the video.
        output_path: path to save the output video.
        reps: list of predicted start and end indices.
        states: list of predicted states.
        step: step size of the predictions.
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f'Failed to open {video_path}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if output_path.endswith('.webm'):
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'vp80'), fps,
                              (width, height))
    else:
        if not output_path.endswith('.mp4'):
            output_path += '.mp4'
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                              (width, height))

    for idx, res in enumerate(np.repeat(states, step)):
        ret, frame = cap.read()
        if not ret:
            break
        count_idx = bisect_left(reps[::2], idx)
        cv2.putText(frame, f'class {res}', (int(width * 0.2), int(height * 0.25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS['red'], 2)
        cv2.putText(frame, f'count {count_idx}', (int(width * 0.25), int(height * 0.5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS['orange'], 2)
        out.write(frame)
    cap.release()
    out.release()


def pred_to_count(preds: List[int], step: int) -> Tuple[int, List[int]]:
    """Convert a list of predictions to a repetition count.
    
    Args:
        preds: list of size total_frames//step in the video. If -1, it means no action.
        step: step size of the predictions.

    Returns:
        A tuple of (repetition count, 
        list of preds of action start and end states, e.g. start_1, end_1, start_2, end_2, ...)

    Note:
        The labels are in order. Because that's how I loaded the data.
        E.g. 0 and 1 represent the start and end of the same action.
        We consider action class as well as state changes.
        I can ensemble a standalone action recognition model if things don't work well.

    Algorithm:
        1. If the state changes, and current and previous state are of the same action,
            and are in order, we count the action. For example, if the state changes from
            0 to 1, or 2 to 3, aka even to odd, we count the action.
        
        It means the model has to capture the presice time of state transition.
        Because the model takes 8 continuous frames as input.
        Or I doubt it will work well. So multiple time scale should be added.
    
    Example:
        >>> preds = [-1, -1, 6, 6, 6, 7, 6, 6, 6, 7, 6, 6, 7, 7, 6, 6, 7, 7, 6, 6, 7, 7, 6, 6, 7, 7, -1]
        >>> pred_to_count(preds, step=8)
        (6, [16, 40, 48, 72, 80, 96, 112, 128, 144, 160, 176, 192])
    """

    count = 0
    reps = []  # start_1, end_1, start_2, end_2, ...
    states: List[int] = []
    prev_state_start_idx = 0
    for idx, pred in enumerate(preds):
        if pred == -1:
            continue
        # if state changed and current and previous state are the same action
        if states and states[-1] != pred:
            if pred % 2 == 1 and states[-1] == pred - 1:
                count += 1
                reps.append(prev_state_start_idx * step)
                reps.append(idx * step)
        states.append(pred)
        prev_state = preds[prev_state_start_idx]
        if pred != prev_state:  # new state, new start index
            prev_state_start_idx = idx

    assert count * 2 == len(reps)
    return count, reps  # len(rep) * step <= len(frames), last not full queue is discarded


def inference_image(model: Union[onnxruntime.InferenceSession, torch.nn.Module],
                    frame: np.ndarray) -> np.ndarray:
    """Inference using image classification model.
    
    Args:
        model: ONNX Runtime session or Pytorch model.
        frame (numpy.ndarray): cv2 read image. Shape (H, W, 3).
    Returns:
        numpy.ndarray: list of prediction scores. Shape (1, num_classes).
    """

    if type(model) == onnxruntime.InferenceSession:
        frame = data_transform(frame).unsqueeze(0).numpy()  # type: ignore
        input_name = model.get_inputs()[0].name
        ort_inputs = {input_name: frame}
        ort_outs = model.run(None, ort_inputs)
        score = ort_outs[0][0]
    else:
        score = model(data_transform(frame).cuda().unsqueeze(0))
        score = score.detach().cpu().numpy()[0]
    print(f'score={score}')
    return score.astype(np.float32)


def count_by_image_model(model: Union[onnxruntime.InferenceSession, torch.nn.Module],
                         video_path: str,
                         ground_truth: Optional[List[int]] = None,
                         video_out_path: Optional[str] = None,
                         pred_out_path: Optional[str] = None,
                         threshold: float = 0.1) -> Tuple[int, List[int]]:
    """Counts repetition int a video, using image classification model.
    
    Args:
        ort_session: ONNX Runtime session.
        video_path: video to be evaluated.
        ground_truth: list, column `reps` in `annotation.csv`.
        If None, metrics will not be calculated.
        output_path: path to save the output video. If None, no video will be saved.
        pred_out_path: path to save the prediction in txt. The scores saved in one line per frame.
        Separated by comma.
        threshold (float): scores below this threshold will be viewed as background.

    Returns:
        Tuple[int, List[int]]: (repetition count, predicted reps).

    Note:
        Voting is used to determine the repetition count.
    """

    print(f'{video_path}')
    cap = cv2.VideoCapture(video_path)

    count = 0
    que: Deque[int] = Deque(maxlen=7)
    states: List[int] = []  # 0 or 1 of length video_length
    scores: List[np.ndarray] = []  # scores without argmax
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        score = inference_image(model, frame)
        scores.append(score)
        que.append(int(score.argmax()))
        states.append(sum(que) >= 4)

    cap.release()

    count, reps = pred_to_count(preds=states, step=7)
    gt_count = len(ground_truth) // 2 if ground_truth else -1
    correct = (abs(count - gt_count) <= 1)
    print(f'count={count} gt_count={gt_count} correct={correct}')
    if pred_out_path:
        save_scores_to_json(scores, pred_out_path, video_path, step=1)
    if video_out_path:
        write_to_video(video_path, video_out_path, reps, states, step=7)
    return count, reps


def inference_video(model: Union[onnxruntime.InferenceSession, torch.nn.Module],
                    inputs: Union[Tensor, np.ndarray],
                    threshold: float = 0.5,
                    transform: Callable = None) -> List[Tuple[int, float]]:
    """Time shift module inference. 8 frames. # TODO: fix doc and inputs

    Args:
        model: ONNX Runtime session or PyTorch.
        inputs (Tensor or np.ndarray): video clip of shape [batch, 8, 3, 224, 224]
        threshold (float): threshold for bbox. # TODO: implement this.

    Returns:
        List[Tuple[int, float]]: list of (class id, score).

    Example:
        >>> inputs = np.zeros([1, 8, 3, 224, 224])
        >>> inference_video(model, inputs)
        [(8: 0.15422), (2: 0.10173), (9: 0.095170), (5: 0.089147), (3: 0.087190)]
    """
    if isinstance(model, onnxruntime.InferenceSession):
        if type(inputs) is not Tensor:
            x = torch.from_numpy(inputs).float()
        else:
            x = inputs.permute(0, 3, 1, 2)
        assert transform is not None
        x = transform(x)
        x = torch.unsqueeze(x, 0)
        input_name = model.get_inputs()[0].name
        ort_outs = model.run(None, {input_name: x.cpu().numpy()})
        score: np.ndarray = ort_outs[0][0]
        pred = list(enumerate(score.tolist()))
    else:  # use mmlab inference
        input_clip = np.array(x)
        score = inference_recognizer(model, input_clip)
        pred = score  # type: ignore
    # print('score', list(score))
    return pred


def count_by_video_model(model: Union[onnxruntime.InferenceSession, torch.nn.Module],
                         video_path: str,
                         ground_truth: Optional[list] = None,
                         video_out_path: Optional[str] = None) -> Tuple[int, List[int]]:
    """Counts repetition in a video, using video classification model.
    
    Args:
        ort_session: ONNX Runtime session. [1, 8, 3, 224, 224]
        video_path: path to the video.
        ground_truth: list of ground truth repetition counts.
        video_out_path: path to save the output video.
    
    Returns:
        Tuple[int, List[int]]: predicted count and reps.
    
    Note:
        The current implementation is not online inference. Because it's in debug mode.
        Will be updated when the accuracy is good enough.
    
    Example::

        >>> count, reps = count_by_video_model(model, video_path, ground_truth=item.reps,
        ...                                    video_out_path=output_path)
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
            input_clip = np.array(input_queue)  # [8, 3, 224, 224]
            pred = inference_video(model, input_clip)
            pred_class = pred[0][0]
            states.append(pred_class)
            input_queue.clear()
        frame_idx += 1
    cap.release()

    count, reps = pred_to_count(preds=states, step=8)
    gt_count = len(ground_truth) // 2 if ground_truth else -1
    correct = (abs(gt_count - count) <= 1)
    print(f'count={count}, gt_count={gt_count}, correct={correct}')
    if video_out_path is not None:
        write_to_video(video_path, video_out_path, reps, states=states, step=8)
    return count, reps


def inference_dataset(model: nn.Module,
                      splits: List[str],
                      out_dir: str,
                      checkpoint: str,
                      person_crop: bool = False) -> None:
    """Inference the RepCount dataset. Save predictions to json for analysis later.
    For video models, predict every 8 frames. Note that the 8 frames are sampled from 16 frames.
    For image models, predict every 1 frame.
    
    Results are saved in format of:
        video_1.score.json::

            {
                "video_name": "video_1.mp4",
                "scores": {
                    0: [0.1, 0.2, 0.3, ...], # frame_idx: score for every classes
                    1: [0.3, 0.2, 0.1, ...],
                },
                "model": "video_model",
                "input_shape": [1, 8, 3, 224, 224],
                "checkpoint: "{checkpoint_path}",
                "total_frames": 100,
                "ground_truth": [10, 40, 40, 70, ...], # reps
                "action": "pull_up",
            }
    
    Args:
        model (nn.Module): mmaction model.
        splits (List[str]): list of splits to inference. ['train', 'val', 'test']
        out_dir (str): output directory. Will create one if not exists.
        checkpoint (str): path to the checkpoint. Just for logging.

    Example::

        # MMaction example
        >>> cfg_path = 'workoutdetector/configs/tsm_MultiActionRepCount_sthv2.py'
        >>> ckpt = 'checkpoints/tsm_video_all.pth'
        >>> model = init_recognizer(cfg_path, ckpt, device='cuda')
        >>> inference_dataset(model, ['val', 'test'], out_dir='out', checkpoint=ckpt)

        # ONNX example
        >>> ckpt = 'checkpoints/rep_12_20220705_220720.onnx'
        >>> model = onnxruntime.InferenceSession(ckpt, providers=['CUDAExecutionProvider'])
        >>> inference_dataset(model, ['train', 'val', 'test'],
        ...                   out_dir='out/tsm_lightning_sparse_sample',
        ...                   checkpoint=ckpt)
        train951.mp4 result saved to out/tsm_lightning_sparse_sample/train951.mp4.score.json
    """

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    data_root = osp.expanduser('~/data/RepCount/')
    helper = RepcountHelper(data_root, osp.join(data_root, 'annotation.csv'))
    data = helper.get_rep_data(splits, action=['all'])
    transform = build_test_transform(person_crop=person_crop)
    print('==> transform:', transform)
    for item in data.values():
        vid = read_video(item.video_path)[0]
        res_dict = dict(
            video_name=item.video_name,
            model='video_model',
            input_shape=[1, 8, 3, 224, 224],
            checkpoint=checkpoint,
            total_frames=len(vid),
            ground_truth=item.reps,
            action=item.class_,
        )
        scores: Dict[int, dict] = dict()
        for i in range(0, len(vid), 8):
            clip = vid[i:i + 16:2]  # sparse sampling
            if len(clip) < 16:
                clip = torch.cat([clip, torch.zeros((8 - len(clip),) + clip.shape[1:])])
            pred = inference_video(model, clip, transform=transform)
            scores[i] = dict((x[0], float(x[1])) for x in pred)
            # print(scores[i])
        res_dict['scores'] = scores
        out_path = os.path.join(out_dir, f'{item.video_name}.score.json')
        json.dump(res_dict, open(out_path, 'w'))
        print(f'{item.video_name} result saved to {out_path}')


def eval_dataset(model: Union[onnxruntime.InferenceSession, torch.nn.Module],
                 action: List[str],
                 split: str,
                 model_type: str = 'video',
                 output_dir: Optional[str] = None,
                 csv_name: str = None,
                 save_video: bool = False,
                 threshold: float = 0.7) -> None:
    """Inference on a dataset test split. Only evaluates count metrics for now.
    The precise repetition metrics will be implemented when I read more papers on it.
    Going to refer to action detection and action segmentation papers for the details.
    
    Args:
        model: ONNX Runtime session or PyTorch model. Supports image and video models.
        action: list of action name.
        split: str, split name.
        model_type: model type. Image or video model.
        output_dir: path to save the output videos and result csv.
        csv_name: name of the csv file. 
        If None, use the default name `eval_count_{model_type}_model.csv`.
        save_video: if True, save the output videos in the output_dir.
        threshold (float):  score smaller than threshold will be viewed as background.
    TODO:
        Implement the repetition metrics.

    Example::

        >>> cfg_path = 'workoutdetector/configs/tsm_MultiActionRepCount_sthv2.py'
        >>> checkpoint = 'checkpoints/tsm_video_all.pth'
        >>> model = init_recognizer(cfg_path, checkpoint, device='cuda')
        >>> actions = ['situp', 'push_up', 'pull_up', 'jump_jack', 'squat', 'front_raise']
        >>> output_dir = 'out/'
        >>> eval_dataset(model,
                     action=actions,
                     split='val',
                     model_type='video',
                     output_dir=output_dir,
                     csv_name=csv_name)

        MAE={mae}, OBO_ACC={obo_acc}, SPLIT={split}, ACTION={action}
        ...
        Saved to: out/tsm_video_all.csv
    """

    data_root = os.path.join(PROJ_ROOT, 'data/RepCount/')
    assert data_root is not None
    helper = RepcountHelper(data_root, REPCOUNT_ANNO_PATH)
    repcount_items = helper.get_rep_data(split=[split], action=action)
    pred_dict = dict()
    for name, item in repcount_items.items():
        assert os.path.exists(item['video_path']), f'{item["video_path"]} not exists'
        if save_video and output_dir is not None:
            assert os.path.isdir(output_dir)
            assert name.endswith('.mp4')
            output_path = os.path.join(output_dir, name)
        else:
            output_path = None
        if model_type == 'video':
            count, reps = count_by_video_model(model,
                                               item.video_path,
                                               ground_truth=item.reps,
                                               video_out_path=output_path)

        elif model_type == 'image':
            count, reps = count_by_image_model(model,
                                               item.video_path,
                                               ground_truth=item.reps,
                                               video_out_path=output_path,
                                               pred_out_path=None,
                                               threshold=threshold)
        else:
            raise ValueError(f'Invalid model type: {model_type}')
        pred_dict[name] = count  # Only implemented count evaluation for now.
    mae, obo_acc, eval_res = helper.eval_count(pred_dict, action=action, split=[split])
    print(f'MAE={mae}, OBO_ACC={obo_acc}, SPLIT={split}, ACTION={action}')
    if output_dir is not None:  # write to csv
        res = []
        for item in eval_res.values():
            dict_ = item.__dict__
            dict_.pop('video_path')
            dict_.pop('frames_path')
            res.append(dict_)
        df = pd.DataFrame.from_dict(res)
        if csv_name is None:
            csv_name = f'eval_count_{model_type}_model.csv'
        if os.path.isfile(csv_name):  # if exists, add timestamp
            csv_name = csv_name.split('.')[0] + '_' + str(time.time()) + '.csv'
        df.to_csv(os.path.join(output_dir, csv_name))
        print(f'Saved to {os.path.join(output_dir, csv_name)}')


def main(args) -> None:
    if args.mmlab:
        cfg_path = os.path.join(
            PROJ_ROOT, 'workoutdetector/configs/tsm_MultiActionRepCount_sthv2.py')
        model = init_recognizer(cfg_path, args.checkpoint, device='cuda')
    else:
        if args.checkpoint.endswith('.pt'):
            model = torch.jit.load(args.checkpoint)
            model.cuda()
        elif args.checkpoint.endswith('.onnx'):
            model = onnxruntime.InferenceSession(
                args.checkpoint,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    if not args.eval and args.video is not None:
        video_path = args.video
        if args.model_type == 'image':
            video_name = os.path.basename(video_path)
            pred_path = osj(os.path.dirname(args.output), f'{video_name}-score.json')
            count_by_image_model(model,
                                 video_path,
                                 ground_truth=[],
                                 video_out_path=args.output,
                                 pred_out_path=pred_path,
                                 threshold=args.threshold)
        elif args.model_type == 'video':
            count_by_video_model(model,
                                 video_path,
                                 ground_truth=[],
                                 video_out_path=args.output)
    elif args.eval:
        CLASSES = ['situp', 'push_up', 'pull_up', 'jump_jack', 'squat', 'front_raise']
        if args.action == 'all':
            action = CLASSES
        else:
            action = [args.action]
        csv_name = args.checkpoint.split('.')[0].split('/')[-1] + '.csv'
        eval_dataset(model,
                     action=action,
                     split=args.split,
                     model_type=args.model_type,
                     output_dir=args.output,
                     csv_name=csv_name)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate RepCount')
    parser.add_argument(
        '-ckpt',
        '--checkpoint',
        help='checkpoint path. Use onnx if ends with .onnx. Use torch script if .pt',
        required=True)
    parser.add_argument('--mmlab', help='use mmlab model', action='store_true')
    parser.add_argument('-i', '--video', help='video path', required=False)
    parser.add_argument('--eval', help='evaluate dataset', action='store_true')
    parser.add_argument('-t', '--threshold', help='threshold', type=float, default=0.5)
    parser.add_argument('-o',
                        '--output',
                        help='video output path. If evaluate dataset, it is output_dir',
                        required=False)
    parser.add_argument('-m',
                        '--model-type',
                        help='inference using image/video model',
                        default='image',
                        choices=['image', 'video'])
    parser.add_argument('-a',
                        '--action',
                        help='action name',
                        default='situp',
                        choices=[
                            'situp', 'push_up', 'pull_up', 'jump_jack', 'squat',
                            'front_raise', 'all'
                        ])
    parser.add_argument('-s',
                        '--split',
                        help='split',
                        default='test',
                        choices=['test', 'train', 'val'])

    args = parser.parse_args(argv)
    return args


if __name__ == '__main__':
    from mmaction.apis import init_recognizer
    from mmaction.apis.inference import inference_recognizer
    example_args = [
        '-ckpt', 'checkpoints/tsm_video_all_20220616.onnx', '--threshold', '0.5',
        '--video', 'data/RepCount/videos/test/stu1_27.mp4'
    ]
    example_dataset = [
        '-ckpt', 'checkpoints/tsm_video_all_20220616.onnx', '--eval', '--output', 'out/',
        '--model-type', 'video', '--action', 'all'
    ]
    example_mmlab = [
        '--mmlab', '-ckpt', 'checkpoints/tsm_video_all.pth', '--eval', '--output', 'out/',
        '--model-type', 'video', '--action', 'pull_up'
    ]
    example_image = [
        '-ckpt=checkpoints/pull-up-image-swin-1x3x224x224.onnx', '--model-type=image',
        '--output=out/', '--action=pull_up', '--eval'
    ]
    # args = parse_args()
    # main(args)

    cfg_path = 'workoutdetector/configs/tsm_MultiActionRepCount_sthv2.py'
    ckpt = 'checkpoints/repcount-12/rep_12_20220705_220720.onnx'
    model = onnxruntime.InferenceSession(ckpt, providers=['CUDAExecutionProvider'])
    # model = init_recognizer(cfg_path, ckpt, device='cuda')
    inference_dataset(model, ['train', 'val', 'test'],
                      out_dir='out/tsm_lightning_sparse_sample',
                      checkpoint=ckpt)
