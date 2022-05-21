train = '/home/umi/projects/WorkoutDetector/data/Countix/rawframes/train.txt'
val = '/home/umi/projects/WorkoutDetector/data/Countix/rawframes/val.txt'

import os
base = '/home/umi/projects/WorkoutDetector/'
traindir = os.path.join(base, 'data/Workouts/rawframes/Countix/train')
valdir = os.path.join(base, 'data/Workouts/rawframes/Countix/val')

import pandas as pd

traindf = pd.read_csv('/home/umi/projects/WorkoutDetector/datasets/Countix/workouts_train.csv')
valdf = pd.read_csv('/home/umi/projects/WorkoutDetector/datasets/Countix/workouts_val.csv')
classes = []
with open('/home/umi/projects/WorkoutDetector/datasets/Countix/classes.txt') as f:
    classes = [line.rstrip() for line in f]

with open(train, 'w') as f:
    for i, row in traindf.iterrows():
        vid = row['video_id']
        label = classes.index(row['class'])
        if os.path.exists(os.path.join(traindir, vid)):
            num_frames = len(os.listdir(os.path.join(traindir, vid)))
            f.write(f'{vid} {num_frames} {label}\n')

with open(val, 'w') as f:
    for i, row in valdf.iterrows():
        vid = row['video_id']
        label = classes.index(row['class'])
        if os.path.exists(os.path.join(valdir, vid)):
            num_frames = len(os.listdir(os.path.join(valdir, vid)))
            f.write(f'{vid} {num_frames} {label}\n')