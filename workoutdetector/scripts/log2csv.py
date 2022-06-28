import pandas as pd
import numpy as np
import os
import re


HOME = '/home/umi/projects/WorkoutDetector'
anno = pd.read_csv(os.path.join(HOME, 'data/RepCount/annotation.csv'))

regex = r'(?:count: (\d+), gt_count: (\d+) )?error: (\d+)\s+error rate: ([.\d]*)'

names = []
errors = []
error_rates = []
counts = []
rep_lists = []
gt_counts = []

with open(os.path.join(HOME, 'exp/tmp.txt'), 'r') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        if line.startswith('/home'):
            name = line.strip().split('/')[-1]
            match = re.findall(regex, lines[idx+1])
            if match:
                count, gt_count, error, error_rate = match[0]
                names.append(name)
                counts.append(count)
                gt_counts.append(gt_count)
                errors.append(error)
                error_rates.append(error_rate)

pd.DataFrame({
    'name': names,
    'count': counts,
    'gt_count': gt_counts,
    'error': errors,
    'error_rate': error_rates,
}).to_csv(os.path.join(HOME, 'exp/tmp.csv'), index=False)
