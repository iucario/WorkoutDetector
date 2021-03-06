from typing import Dict, List, Optional, OrderedDict, Tuple, Union
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from workoutdetector.utils import pred_to_count, to_softmax
import os


def obo_mae(preds: List[int],
            targets: List[int],
            ratio: bool = True) -> Union[Tuple[float, int], Tuple[float, float]]:
    """Evaluate count prediction. By mean absolute error and off-by-one error."""

    mae = 0.0
    off_by_one = 0.0
    for pred, target in zip(preds, targets):
        mae += abs(pred - target)
        off_by_one += (abs(pred - target) == 1)
    if ratio:
        return mae / len(preds), off_by_one / len(preds)
    else:
        return mae / len(preds), off_by_one


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


def analyze_count(csv: str, out_csv: Optional[str]) -> None:
    """Input a csv file, analyze results
    
    Args:
        csv (str): path to csv file, with columns:
            `,name,gt_count,pred_count,gt_rep,pred_rep,split,action`
        out_csv (str or None): path to save output csv file, with columns:
            `,action,split,mae,obo_acc,total,avg_count`
    Example:
        >>> in_csv = 'out/tsm_lightning_sparse_sample_eval.csv'
        >>> out_csv = in_csv.replace('.csv', '_meta.csv')
        >>> analyze_count(in_csv, out_csv)
    """

    df = pd.read_csv(csv, index_col='name')
    actions = df.action.unique()
    splits = df.split.unique()
    out = []
    split_out: Dict[str,
                    dict] = dict((sp, {
                        'mae': 0,
                        'obo': 0,
                        'total': 0,
                        'avg_count': 0.0
                    }) for sp in splits)
    for split in splits:
        for action in actions:
            df_action = df.loc[(df.action == action) & (df.split == split)]
            gt_count = df_action.gt_count.values
            pred_count = df_action.pred_count.values
            mae, obo = obo_mae(pred_count, gt_count, ratio=False)
            avg_count = np.mean(gt_count)
            out.append([action, split, mae, obo, len(df_action), avg_count])
            split_out[split]['mae'] += int(mae * len(df_action))
            split_out[split]['obo'] += int(obo)
            split_out[split]['total'] += len(df_action)
            split_out[split]['avg_count'] += gt_count.sum()

    df_out = pd.DataFrame(
        out, columns=['action', 'split', 'mae', 'obo_acc', 'total', 'avg_count'])
    # all actions per split
    for split in splits:
        print(f'{split}: {split_out[split]}')
        total = split_out[split]['total']
        row = pd.DataFrame({
            'action': 'all',
            'split': split,
            'mae': split_out[split]['mae'] / total,
            'obo_acc': split_out[split]['obo'],
            'total': total,
            'avg_count': split_out[split]['avg_count'] / total
        }, index=[0])
        df_out = df_out.append(row, ignore_index=True)

    if out_csv:
        df_out.to_csv(out_csv)
    print(df_out)


def main(json_dir: str,
         anno_path: str,
         out_csv: Optional[str],
         softmax: bool = False) -> None:
    """Evaluates obo accuracy and mean absolute error.
    
    Args:
        json_dir (str): path to dir containing json files with name
            `{video_name}.json
        anno_path (str): path to annotation file
        out_csv (str or None): path to save output csv file, with columns:
            `,name,gt_count,pred_count,gt_rep,pred_rep,split,action`
        softmax (bool): whether to apply softmax
    Example:
        >>> json_dir = 'out/tsm_lightning_sparse_sample'
        >>> anno_path = 'data/RepCount/annotation.csv'
        >>> out_csv = 'out/tsm_lightning_sparse_sample_eval.csv'
        >>> main(json_dir, anno_path, out_csv)
        Done. csv file saved to out/tsm_lightning_sparse_sample_eval.csv
        =====Mean absolute error: 4.0141, OBO acc: 0.2293=====
    """

    threshold = 0.5
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
        scores: Dict[str, Dict[str, float]] = json_data['scores']
        if softmax:
            for k in scores.keys():
                scores[k] = to_softmax(scores[k])
        gt: List[int] = anno.loc[video_name]['reps']
        gt_count = anno.loc[video_name]['count'].astype(int)
        pred = []
        for v in scores.values():
            class_id, score = max(v.items(), key=lambda x: x[1])
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
    df = pd.DataFrame(out,
                      columns=[
                          'name', 'gt_count', 'pred_count', 'gt_rep', 'pred_rep', 'split',
                          'action'
                      ])
    if out_csv:
        df.to_csv(out_csv)
        print(f'Done. csv file saved to {out_csv}')
    print(f'=====Mean absolute error: {mae:.4f}, OBO acc: {obo:.4f}=====')


if __name__ == '__main__':
    d = 'tsm_lightning_sparse_sample'
    json_dir = f'out/{d}'
    anno_path = 'data/RepCount/annotation.csv'
    out_csv = f'out/{d}.csv'
    # main(json_dir, anno_path, out_csv, softmax=True)
    analyze_count(out_csv, out_csv.replace('.csv', '_meta.csv'))
