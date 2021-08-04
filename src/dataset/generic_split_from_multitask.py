"""Partition a generic (processed) csv and write out the multitask indices `/data/splits`"""
import os
import numpy as np
import pandas as pd

import constants


DATASET_VERSION = 'v1'

in_path = os.path.join(constants.SPLIT_DIR, f'split_multitask_{DATASET_VERSION}.npz')
df_path = os.path.join(constants.DATAFRAME_DIR, f'dataset_{DATASET_VERSION}.csv')

def print_split_percent(train, val, test):
  train, val, test = len(train), len(val), len(test)
  total = train + val + test
  print("\tTrain: {:.02f}; Val: {:.02f}; Test: {:.02f}".format(train / total, val / total, test/total))

def get_train_val_test_split(df, multi_splits, trait):
  print("\n", trait)
  df['col_index'] = range(0, len(df))
  print_split_percent(df.loc[multi_splits['train']],
                      df.loc[multi_splits['val']],
                      df.loc[multi_splits['test']])

  train = df.loc[multi_splits['train']].dropna(subset=[trait])
  val = df.loc[multi_splits['val']].dropna(subset=[trait])
  test = df.loc[multi_splits['test']].dropna(subset=[trait])
  print_split_percent(train, val, test)

  return train['col_index'].values, val['col_index'].values, test['col_index'].values


def save(trait, train, val, test):
  print("Trait:", trait)
  print("\tTrain size:", len(train))
  print("\tVal size:", len(val))
  print("\tTest size:", len(test))

  print(os.path.join(constants.SPLIT_DIR, f'split_{trait}_{DATASET_VERSION}.npz'))
  np.savez(os.path.join(constants.SPLIT_DIR, f'split_{trait}_{DATASET_VERSION}.npz'), train=train, val=val, test=test)


print("Processing: ", DATASET_VERSION)
df = pd.read_csv(df_path)
df['col_index'] = range(0, len(df))

multi_splits = dict(np.load(in_path))

for trait in constants.TRAITS:
  np.random.seed(42)
  train, val, test = get_train_val_test_split(df, multi_splits, trait)
  save(trait, train, val, test)
# train, val, test = get_train_val_test_split(in_path)
#
# print("\tTrain size:", len(train))
# print("\tVal size:", len(val))
# print("\tTest size:", len(test))


# print(multi_splits)
