import os
import pandas as pd
from config import BASEDIR
import shutil

clips = os.listdir(os.path.join(BASEDIR, 'data', 'clips'))

df = pd.read_csv(os.path.join(BASEDIR, 'data/anno.csv'))


def split_train_test(df, train_size=0.8):
    train = df.sample(frac=train_size, random_state=0)
    test = df.drop(train.index)
    return train, test


# exist_files = [x[:11] for x in os.listdir(os.path.join(BASEDIR, 'data/clips'))]
# df = df[df['vid'].isin(exist_files)]

# train, test = split_train_test(df)
train = pd.read_csv(os.path.join(BASEDIR, 'data/train.csv'))
test = pd.read_csv(os.path.join(BASEDIR, 'data/val.csv'))


def write_csv(df, path):
    df.to_csv(path, index=False)


# write_csv(df, os.path.join(BASEDIR, 'data/anno.csv'))
# write_csv(train, os.path.join(BASEDIR, 'data/train.csv'))
# write_csv(test, os.path.join(BASEDIR, 'data/val.csv'))


def copy_files():
    train.apply(lambda x: shutil.copy(
        os.path.join(BASEDIR, 'data/clips', f"{x['vid']}_{x['start_sec']}.mp4"),
        os.path.join(BASEDIR, 'data/train')), axis=1)

    test.apply(lambda x: shutil.copy(os.path.join(
        BASEDIR, 'data/clips', f"{x['vid']}_{x['start_sec']}.mp4"),
        os.path.join(BASEDIR, 'data/val')), axis=1)


copy_files()
