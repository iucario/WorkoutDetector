from typing import List, Tuple
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from workoutdetector.utils.inference_count import count_by_video_model


def obo_mae(preds: List[int], targets: List[int]) -> Tuple[float, float]:
    """Evaluate count prediction. By mean absolute error and off-by-one error."""

    mae = 0.0
    off_by_one = 0.0
    for pred, target in zip(preds, targets):
        mae += abs(pred - target)
        off_by_one += (abs(pred - target) == 1)
    return mae / len(preds), off_by_one / len(preds)


def eval_count(preds: List[int], gt: List[int]):
    """Evaluates counting accuracy
    
    Args:
        preds (list of int): predicted reps. [start_1, end_1, ...]
        gt (list of int): ground truth reps
    Returns:
        Tuple[float]: Recall and precision
    Metrics:
        - OBO (off by one) accuracy
        - MAE: mean averaged error
        * True positive: count when repetition occurs
        * False positive: count when no repetition occurs
        - Recall: TP / predicted counts
        - Precision: TP / ground truth counts
    Example::

        >>> preds = [1, 3, 5, 7]
        >>> gt = [0, 2, 4, 6]
        >>> eval_count(preds, gt)
        (1.0, 1.0)

        >>> preds = [1, 3, 5, 7]
        >>> gt = [0, 5, 7, 9]
        >>> eval_count(preds, gt)
        (0.5, 0.5)
    """
    # TODO: how to define true positive and false positive?
    pass

