import json
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from workoutdet.predict import pred_to_count
from workoutdet.utils import to_softmax


def obo_mae(
        preds: List[int],
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


def count_stats(csv: str, out_csv: bool = True) -> None:
    """Input a csv file, analyze results. Accuracy, etc.
    
    Args:
        csv (str): path to csv file, with columns:
            `,name,gt_count,pred_count,gt_rep,pred_rep,split,action`
        out_csv (bool): save results to path `{the_input_csv_name}_meta.csv`,
            default True
    Example:
        >>> count_stats('out/tsm_sparse_sample_eval.csv')
    """

    df = pd.read_csv(csv, index_col='name')
    actions = df.action.unique()
    splits = df.split.unique()
    out = []
    split_out: Dict[str, dict] = dict((sp, {
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
        out,
        columns=['action', 'split', 'MAE', 'OBO_acc', 'total', 'avg_count'])
    # all actions per split
    for split in splits:
        print(f'{split}: {split_out[split]}')
        total = split_out[split]['total']
        df_out.loc[len(df_out)] = [
            'all', split, split_out[split]['mae'] / total,
            split_out[split]['obo'], total,
            split_out[split]['avg_count'] / total
        ]
    df_out['OBO_acc'] = df_out['OBO_acc'].astype(int)
    if out_csv:
        df_out.to_csv(csv[:-4].replace('.csv', '_meta.csv'))
    print(df_out.to_latex(index=False))


def major_vote(pred: List[int], window: int) -> List[int]:
    """Return the most frequent predicted labels in non-overlapping windows."""
    pred_smooth = []
    for i in range(0, len(pred), window):
        counter = Counter(pred[i:min(i + window, len(pred))])
        pred_smooth.append(counter.most_common(1)[0][0])
    return pred_smooth


def load_json(fpath, softmax: bool = True) -> Tuple[dict, str]:
    with open(fpath) as fp:
        json_data = json.load(fp)
    # scores: {frame_index: {class_1: score}, {class_2: score}}
    scores: Dict[str, Dict[str, float]] = json_data['scores']
    if softmax:
        for k in scores.keys():
            scores[k] = to_softmax(scores[k])
    return scores, json_data['action']


def infer(scores: dict, threshold: float, window=10) -> List[int]:
    """Naive inference by counting state transition"""
    pred = []
    for v in scores.values():
        class_id, score = max(v.items(), key=lambda x: x[1])
        if score >= threshold:
            pred.append(int(class_id))
        else:
            pred.append(-1)
    # smooth predictions
    pred = major_vote(pred, window)
    return pred


def main(json_dir: str,
         anno_path: str,
         out_csv: Optional[str] = None,
         softmax: bool = True,
         threshold: float = 0.5,
         stride: int = 1,
         step: int = 2,
         window: int = 1) -> None:
    """Evaluates obo accuracy and mean absolute error.
    
    Args:
        json_dir (str): path to dir containing json files with name
            `{video_name}.json
        anno_path (str): path to annotation file
        out_csv (str or None): path to save output csv file, with columns:
            `,name,gt_count,pred_count,gt_rep,pred_rep,split,action`
        softmax (bool): whether to apply softmax
        threshold (float): frames with score under threshold will be skipped
        stride (int): stride of each evaluation
        step (int): step of frames to evaluate
        window (int): majority vote window size
    Example:
        >>> json_dir = 'out/tsm_sparse_sample'
        >>> anno_path = 'data/RepCount/annotation.csv'
        >>> out_csv = 'out/tsm_sparse_sample_eval.csv'
        >>> main(json_dir, anno_path, out_csv)
        Done. csv file saved to out/tsm_sparse_sample_eval.csv
        =====Mean absolute error: 4.0141, OBO acc: 0.2293=====
    """

    files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    anno = pd.read_csv(anno_path, index_col='name')
    out = []
    preds = []
    gts = []

    for f in files:
        video_name = f.split('.')[0] + '.mp4'
        if not video_name in anno.index:
            print(f'{video_name} not in annotation')
            continue
        # print(f'Evaluating {video_name}')
        scores, action = load_json(os.path.join(json_dir, f), softmax=softmax)
        gt: List[int] = anno.loc[video_name]['reps']
        gt_count = anno.loc[video_name]['count'].astype(int)
        pred = infer(scores, threshold, window)
        pred_count, pred_rep = pred_to_count(pred,
                                             stride=stride * window,
                                             step=step)
        preds.append(pred_count)
        gts.append(gt_count)
        split = anno.loc[video_name]['split']
        out.append(
            [video_name, gt_count, pred_count, gt, pred_rep, split, action])
    mae, obo = obo_mae(preds, gts)
    df = pd.DataFrame(out,
                      columns=[
                          'name', 'gt_count', 'pred_count', 'gt_reps',
                          'pred_reps', 'split', 'action'
                      ])
    if out_csv:
        df.to_csv(out_csv)
        print(f'Done. csv file saved to {out_csv}')
    print(f'=====Mean absolute error: {mae:.4f}, OBO acc: {obo:.4f}=====')


if __name__ == '__main__':
    d = 'acc_0.841_epoch_26_20220711-191616_1x1'
    json_dir = f'out/{d}'
    anno_path = 'datasets/RepCount/annotation.csv'
    window = 10
    threshold = 0.5
    out_csv = f'out/{d}_window_{window}_threshold_{threshold}.csv'
    main(json_dir,
         anno_path,
         out_csv,
         softmax=True,
         window=window,
         stride=1,
         step=1,
         threshold=threshold)
    count_stats(out_csv)
