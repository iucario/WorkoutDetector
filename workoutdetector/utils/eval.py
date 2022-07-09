from typing import Dict, List, OrderedDict, Tuple
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from workoutdetector.utils.inference_count import count_by_video_model, pred_to_count
import os


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


def main(json_dir: str, anno_path: str, out_csv: str) -> None:
    """Evaluates obo accuracy and mean absolute error.
    
    Args:
        json_dir (str): path to dir containing json files with name
            `{video_name}.json
        anno_path (str): path to annotation file
        out_csv (str): path to save output csv file
    """

    threshold = 0.1
    step = 8
    files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    anno = pd.read_csv(anno_path, index_col='name')
    out = []
    preds = []
    gts = []

    for f in files:
        video_name = f.split('.')[0] + '.mp4'
        with open(os.path.join(json_dir, f)) as fp:
            json_data = json.load(fp)
        # scores: {frame_index: {class_1: score}, {class_2: score}}
        scores: Dict[str, OrderedDict[str, float]] = json_data['scores']
        gt: List[int] = anno.loc[video_name]['reps']
        gt_count = anno.loc[video_name]['count'].astype(int)
        pred = []
        for v in scores.values():
            class_id, score = list(v.items())[0]
            if score >= threshold:
                pred.append(int(class_id))
            else:
                pred.append(-1)
        pred_cout, pred_rep = pred_to_count(pred, step=step)
        preds.append(pred_cout)
        gts.append(gt_count)
        split = anno.loc[video_name]['split']
        action = json_data['action']
        out.append([video_name, gt_count, pred_cout, gt, pred_rep, split, action])

    mae, obo = obo_mae(preds, gts)
    df = pd.DataFrame(
        out, columns=['name', 'gt_count', 'pred_count', 'gt_rep', 'pred_rep', 'split', 'action'])
    df.to_csv(out_csv)
    print(f'Done. csv file saved to {out_csv}')
    print(f'=====Mean absolute error: {mae:.4f}, OBO acc: {obo:.4f}=====')


if __name__ == '__main__':
    json_dir = 'out/tsm_lightning_sparse_sample'
    anno_path = 'data/RepCount/annotation.csv'
    out_csv = 'out/tsm_lightning_sparse_sample_eval.csv'
    main(json_dir, anno_path, out_csv)
