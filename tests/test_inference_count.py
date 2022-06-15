from WorkoutDetector.utils.inference_count import *


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

    x1 = [0] * 10 + [1, 1, 0, 0, 0, 0]
    y1 = (1, [8 * 11 - 1])
    assert pred_to_count(pred_to_index(step=8, preds=x1)) == y1
