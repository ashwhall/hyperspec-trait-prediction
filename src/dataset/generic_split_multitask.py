"""Partition a generic (processed) csv and write out the multitask indices `/data/splits`"""
import os
import numpy as np
import pandas as pd

import constants


DATASET_VERSION = 'v1'

in_path = os.path.join(constants.DATAFRAME_DIR, f'dataset_{DATASET_VERSION}.csv')
out_path = os.path.join(constants.SPLIT_DIR, f'split_multitask_{DATASET_VERSION}.npz')

def get_train_val_test_split(data_file):
    df = pd.read_csv(data_file)

    df['col_index'] = range(0, len(df))

    train=df.sample(frac=constants.SPLIT_RATIOS[0])
    val = df.drop(train.index).sample(frac=(constants.SPLIT_RATIOS[1] / (1 - constants.SPLIT_RATIOS[0])))
    test=df.drop(train.index).drop(val.index)

    return train['col_index'].values, val['col_index'].values, test['col_index'].values



np.random.seed(42)
print("Processing: ", in_path)
train, val, test = get_train_val_test_split(in_path)
data = pd.read_csv(in_path)

print("\tTrain size:", len(train))
print("\tVal size:", len(val))
print("\tTest size:", len(test))


np.savez(out_path, train=train, val=val, test=test)
