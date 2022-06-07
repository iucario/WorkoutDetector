import pandas as pd
import yaml
import os


config = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), '../utils/config.yml')))
base = config['proj_root']

train = os.path.join(base, 'data/Countix/rawframes/train.txt')
val = os.path.join('data/Countix/rawframes/val.txt')


traindir = os.path.join(base, 'data/Workouts/rawframes/Countix/train')
valdir = os.path.join(base, 'data/Workouts/rawframes/Countix/val')


traindf = pd.read_csv(
    os.path.join(base, 'datasets/Countix/workouts_train.csv'))
valdf = pd.read_csv(
    os.path.join(base, 'datasets/Countix/workouts_val.csv'))
classes = []
with open(os.path.join(base, 'datasets/Countix/classes.txt')) as f:
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
