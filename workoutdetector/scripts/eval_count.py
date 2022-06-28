import os
import os.path as osp
from os.path import join as osj
import re
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
from workoutdetector.trainer import LitImageModel
from workoutdetector.settings import PROJ_ROOT, REPCOUNT_ANNO_PATH

anno = pd.read_csv(REPCOUNT_ANNO_PATH)


def func_1(df):

    total = len(df)
    offbyone = sum(df['error'] == 0) + sum(df['error'] == 1)
    print(f'offbyone rate = {offbyone}/{total}')

    # errors for each class
    df['class_'] = df.merge(anno, on='name', how='left')['class_']

    cls_obo = {}

    for class_ in sorted(df['class_'].unique()):
        print(class_)
        c = df[df['class_'] == class_]
        class_obo = sum(c['error'] == 0) + sum(c['error'] == 1)
        print(f'offbyone rate = {class_obo}/{len(c)}')
        cls_obo[class_] = class_obo / len(c)

    #  plot cls_obo
    plt.figure(figsize=(5, 3), dpi=300)
    plt.xlim(0, 1)
    plt.barh(list(cls_obo.keys()), list(cls_obo.values()))
    plt.ylabel('class')
    plt.xlabel('off by one rate')
    plt.title('off by one rate for each class')
    plt.tight_layout()
    plt.savefig('cls_obo.png', dpi=300)
    plt.show()


def func2() -> pd.DataFrame:
    names = []
    errors = []
    counts = []
    gt_counts = []

    with open('/home/umi/projects/WorkoutDetector/exp/image_model_eval.txt') as f:
        lines = f.readlines()
        for x, y in zip(lines[0::2], lines[1::2]):
            x = x.strip()
            y = y.strip()
            name = x.split('/')[-1]
            pattern = r'error=(\d+) count=(\d+) gt_count=(\d+)'
            error, count, gt_count = re.findall(pattern, y)[0]
            print(f'{name} error={error} count={count} gt_count={gt_count}')
            names.append(name)
            errors.append(int(error))
            counts.append(int(count))
            gt_counts.append(int(gt_count))

    df = pd.DataFrame({
        'name': names,
        'count': counts,
        'gt_count': gt_counts,
        'error': errors,
        'error_rate': [x / max(1, y) for x, y in zip(errors, counts)]
    })
    return df


df = pd.read_csv('/home/umi/projects/WorkoutDetector/exp/image_model_eval.csv')

# df['class_'] = df.merge(anno, on='name', how='left')['class_']
# df = pd.read_csv('/home/umi/projects/WorkoutDetector/exp/tmp.csv')

func_1(df)