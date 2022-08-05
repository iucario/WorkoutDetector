import json
from typing import List, Tuple
import numpy as np
import os
from hmmlearn import hmm
from workoutdet.evaluate import major_vote
from workoutdet.predict import pred_to_count
from workoutdet.data import get_rep_data, reps_to_label, FeatureDataset
import pandas as pd


def hmm_infer(model: hmm.GaussianHMM,
              x: np.ndarray,
              gt_reps: List[int],
              class_idx: int,
              window: int = 1) -> Tuple[int, int, float, List[int]]:
    """Predicts repetitions from one video using HMM.
       Set class_idx to 0 to predict 3 classes.
    """
    gt_labels = reps_to_label(gt_reps, len(x),
                              class_idx=class_idx)  # set to 0 for 3 states
    y = model.predict(x)
    acc = (y == gt_labels[:len(y)]).sum() / len(gt_labels)
    gt_count = len(gt_reps) // 2
    pred_smoothed = major_vote(y - 1, window=window)  # minus 1 because no-class is 0
    count, reps = pred_to_count(pred_smoothed, stride=1 * 1, step=1)
    diff = abs(count - gt_count)
    obo = 1 if (diff <= 1) else 0
    return obo, diff, acc, reps


def hmm_eval_subset(model: hmm.GaussianHMM,
                    split: str,
                    action: str,
                    anno_path: str,
                    data_root: str,
                    json_dir: str,
                    template: str = '{}.stride_1_step_1.json') -> dict:
    test_data = list(get_rep_data(anno_path, data_root, [split], [action]).values())
    total_obo, total_err, total_acc, gt_total_count = 0, 0, 0, 0
    for item in test_data:
        test_x = []
        js = json.load(open(os.path.join(json_dir, template.format(item.video_name))))
        for i, v in js['scores'].items():
            test_x.append(np.array(list(v.values())))
        test_x = np.array(test_x)

        obo, err, acc, reps = hmm_infer(model, test_x, item.reps, 0, window=1)
        total_obo += obo
        total_err += err
        total_acc += acc
        gt_total_count += len(item.reps) // 2

    err_rate = total_err / len(test_data)
    acc_rate = total_acc / len(test_data)
    return dict(split=split,
                action=action,
                OBO=total_obo,
                num_videos=len(test_data),
                MAE=err_rate,
                avg_count=gt_total_count / len(test_data),
                hmm_acc=acc_rate)


def hmm_train(action, feat_ds):

    print(action, feat_ds.x.shape, 'num y', np.unique(feat_ds.y))
    # softmax_x = F.softmax(torch.from_numpy(feat_ds.x), dim=1).numpy()
    transmat, pi, means, cov = feat_ds.hmm_stats(feat_ds.x.numpy(), np.array(feat_ds.y),
                                                 'full')
    # print(transmat, pi, means, cov)
    n_states = len(np.unique(feat_ds.y))
    model = hmm.GaussianHMM(n_components=n_states, n_iter=300, covariance_type='full')
    model.transmat_ = transmat
    model.startprob_ = pi.T
    model.means_ = means
    model.covars_ = cov
    # softmax_x = F.softmax(torch.from_numpy(feat_ds.x.squeeze(1)), dim=1)
    # model.fit(feat_ds.x.numpy())
    return model


def hmm_eval(anno_path: str, json_dir: str) -> None:

    result = []
    classes = ['situp', 'push_up', 'pull_up', 'jump_jack', 'squat', 'front_raise']

    for action in classes:
        feat_ds = FeatureDataset(json_dir,
                                 anno_path,
                                 split='train',
                                 action=action,
                                 window=1,
                                 stride=1)
        model = hmm_train(action, feat_ds)
        for split in ['train', 'val', 'test']:
            result.append(hmm_eval_subset(model, split, action))

    df = pd.DataFrame(result)
    df_train = df[df.split == 'train']
    df_val = df[df.split == 'val']
    df_test = df[df.split == 'test']
    all_obo = df.groupby('split').OBO.sum()
    train_mae = (df_train.MAE * df_train.num_videos).sum() / df_train.num_videos.sum()
    val_mae = (df_val.MAE * df_val.num_videos).sum() / df_val.num_videos.sum()
    test_mae = (df_test.MAE * df_test.num_videos).sum() / df_test.num_videos.sum()
    all_num = df.groupby('split').num_videos.sum()
    df_summary = pd.DataFrame({
        'split': ['train', 'val', 'test'],
        'MAE': [train_mae, val_mae, test_mae],
        'OBO': [all_obo['train'], all_obo['val'], all_obo['test']],
        'num_videos': [all_num['train'], all_num['val'], all_num['test']]
    })
    print(df_summary.to_latex(index=False))
    df.to_csv('hmm_result.csv')
    print(df.to_latex(index=False))


if __name__ == '__main__':
    anno_path = 'datasets/RepCount/annotation.csv'
    json_dir = 'json'
    hmm_eval(anno_path, json_dir)