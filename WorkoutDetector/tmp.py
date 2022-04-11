import os
import pandas as pd
from config import BASEDIR

df = pd.read_csv(os.path.join(BASEDIR, 'data/anno.csv'))


def split_train_test(df, train_size=0.8):
    train = df.sample(frac=train_size, random_state=0)
    test = df.drop(train.index)
    return train, test


train, test = split_train_test(df)
print(train.head())
print(test.head())

print(train.shape)
print(test.shape)
print('intersections', train[train.vid.isin(test.vid.unique())])
