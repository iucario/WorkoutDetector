from typing import List

from pytest import fixture
from workoutdetector.utils.inference_count import eval_dataset, pred_to_count


# TODO: load annotation.csv
def test_pred_to_count():
    """Every 8 frames is a prediction"""
    step = 8
    x1 = [0] * 10 + [1, 1, 0, 0, 0, 0]
    y_count = 1
    y_reps = [0, 10]
    assert pred_to_count(stride=step, preds=x1) == (y_count, [x * step for x in y_reps])

    x2 = [0, 0, 2, 2, 2, 5, 5, 5, 5, 6, 6, 9, 9, 9]
    y2_count = 0
    y2_reps = []
    assert pred_to_count(stride=step, preds=x2) == (y2_count, y2_reps)

    x3 = [-1, -1, -1, 1, 1, 2, 3, 2, 3, 2, 3, 3, 3, 0, -1, -1]
    y3_count = 3
    y3_reps = [5, 6, 7, 8, 9, 10]
    assert pred_to_count(stride=step, preds=x3) == (y3_count, [x * step for x in y3_reps])

    x4 = [6, 6, 6, 7, 7, 8, 7, 6, 6, 7]
    y4_count = 2
    y4_reps = [0, 3, 7, 9]
    assert pred_to_count(stride=step, preds=x4) == (y4_count, [x * step for x in y4_reps])

    x5 = [
        -1, -1, 9, 9, 8, -1, -1, -1, -1, -1, -1, 6, 6, 7, 6, 6, 7, 6, 6, 7, -1, -1, -1,
        -1, -1, -1, -1
    ]
    y5_count = 3
    pred_count, pred_rep = pred_to_count(preds=x5, stride=8)
    assert pred_count == y5_count

    x6 = [
        2, 3, 3, 2, 3, 3, 3, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2, 3, 3, 2, 2,
        3, 3, 2, 2, 3, 3, 2, 2, 3, 3, -1
    ]
    y5_count = 10
    y5_rep = [
        0, 8, 24, 32, 56, 64, 80, 96, 112, 128, 144, 160, 176, 184, 200, 216, 232, 248,
        264, 280
    ]
    assert pred_to_count(preds=x6, stride=8) == (y5_count, y5_rep)
