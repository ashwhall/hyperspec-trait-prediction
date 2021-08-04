"""Partition a generic (processed) csv and write out the indices for each trait to `/data/splits`"""
import os
import numpy as np
import pandas as pd

import constants


DATASET_VERSION = 'v1'

in_path = os.path.join(constants.DATAFRAME_DIR, f'dataset_{DATASET_VERSION}.csv')


def get_train_val_test_split(df, trait):
  if trait not in df.columns:
    return [], [], []

  filtered_trait_NA = df[df[trait].notnull()]

  train=filtered_trait_NA.sample(frac=constants.SPLIT_RATIOS[0])
  val = filtered_trait_NA.drop(train.index).sample(frac=(constants.SPLIT_RATIOS[1] / (1 - constants.SPLIT_RATIOS[0])))
  test=filtered_trait_NA.drop(train.index).drop(val.index)

  return train['col_index'].values, val['col_index'].values, test['col_index'].values


def save(trait, train, val, test):
  print("Trait:", trait)
  print("\tTrain size:", len(train))
  print("\tVal size:", len(val))
  print("\tTest size:", len(test))

  print(os.path.join(constants.SPLIT_DIR, f'split_{trait}_{DATASET_VERSION}.npz'))
  np.savez(os.path.join(constants.SPLIT_DIR, f'split_{trait}_{DATASET_VERSION}.npz'), train=train, val=val, test=test)

df = pd.read_csv(in_path)
df['col_index'] = range(0, len(df))
for trait in constants.TRAITS:
  np.random.seed(42)
  print("Processing:", in_path, trait)
  train, val, test = get_train_val_test_split(df, trait)
  save(trait, train, val, test)
print("Done!")



