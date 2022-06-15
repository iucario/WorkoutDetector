from WorkoutDetector.utils.inference_count import *


def test_pred_to_count():
    x1 = [0] * 10 + [1, 1, 0, 0, 0, 0]
    y1 = (1, 10)
    assert pred_to_count(x1) == y1
