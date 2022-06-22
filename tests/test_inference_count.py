from typing import List

from pytest import fixture
from workoutdetector.utils.inference_count import infer_dataset, pred_to_count, parse_args, main


# TODO: load annotation.csv
def test_pred_to_count():
    """Every 8 frames is a prediction"""
    step = 8
    x1 = [0] * 10 + [1, 1, 0, 0, 0, 0]
    y_count = 1
    y_reps = [0, 10]
    assert pred_to_count(step=step, preds=x1) == (y_count, [x * step for x in y_reps])

    x2 = [0, 0, 2, 2, 2, 5, 5, 5, 5, 6, 6, 9, 9, 9]
    y2_count = 0
    y2_reps = []
    assert pred_to_count(step=step, preds=x2) == (y2_count, y2_reps)

    x3 = [-1, -1, -1, 1, 1, 2, 3, 2, 3, 2, 3, 3, 3, 0, -1, -1]
    y3_count = 3
    y3_reps = [5, 6, 7, 8, 9, 10]
    assert pred_to_count(step=step, preds=x3) == (y3_count, [x * step for x in y3_reps])

    x4 = [6, 6, 6, 7, 7, 8, 7, 6, 6, 7]
    y4_count = 2
    y4_reps = [0, 3, 7, 9]
    assert pred_to_count(step=step, preds=x4) == (y4_count, [x * step for x in y4_reps])



def test_infer_dataset():
    args_video = [
        '--onnx', 'checkpoints/tsm_video_all_20220616.onnx', '--threshold', '0.5',
        '--video', 'data/RepCount/videos/test/stu1_27.mp4'
    ]
    args_dataset = [
        '--onnx', 'checkpoints/tsm_video_all_20220616.onnx', '--eval', '--output', 'exp/',
        '--model-type', 'video', '--action', 'pull_up'
    ]
    args_mmlab = [
        '--mmlab', '-ckpt', 'checkpoints/tsm_video_all.pth', '--eval', '--output', 'exp/',
        '--model-type', 'video', '--action', 'pull_up'
    ]
    args = parse_args(args_mmlab)
    main(args)