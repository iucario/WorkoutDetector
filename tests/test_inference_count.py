from WorkoutDetector.utils.inference_count import pred_to_count
from typing import List


def pred_to_index(step: int, preds: List[int]) -> List[int]:
    """Convert preds to index
    
    Args:
        step: int, the step of model's input frames
        preds: List[int], the predictions by the model
    
    Returns:
        List[int], the true index of each frame
    """

    ret = []
    for p in preds:
        ret += [p] * step
    return ret


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