import sys
import os
import re

import numpy as np
import pandas as pd
sys.path.append(os.path.abspath('..'))
try:
  import dataset.constants as constants
except ImportError:
  import constants



def build_file_path_dicts():
  """Build a dict of { version: path } pairs. E.g. { v1: 'dataset_v1.csv' }"""
  file_paths = os.listdir(constants.DATAFRAME_DIR)
  file_paths = {
    re.search('_(.*)\.', d).group(1): os.path.join(constants.DATAFRAME_DIR, d)
    for d in file_paths
  }
  return file_paths

FILE_PATHS = build_file_path_dicts()
MULTITASK_SPLIT_PATHS = {
  d: os.path.join(constants.SPLIT_DIR, f'split_multitask_{d}.npz')
  for d in FILE_PATHS.keys()
}

def load_trait_splits(trait, dataset_version):
  with np.load(os.path.join(constants.SPLIT_DIR, f'split_{trait}_{dataset_version}.npz')) as splits:
    all_split_indices = {}
    for k, v in splits.items():
      all_split_indices[k] = v
  return all_split_indices


def fix_missing_traits(df):
  for trait in constants.TRAITS:
    if trait not in df.columns:
      df[trait] = np.nan
  return df

def fix_missing_values(df, lower_freq, upper_freq):
  ordered_col_names = []
  for v in np.arange(lower_freq, upper_freq+1):
    col_name = f'Wave_{v}'
    ordered_col_names.append(col_name)
    if col_name not in df.columns:
      df[col_name] = 0

  # Sort columns by name, so taking wave_lower:wave_upper includes all waves inbetween
  non_wave_col_names = [c for c in df.columns if c not in ordered_col_names]
  ordered_col_names = non_wave_col_names + ordered_col_names

  df = df.reindex(ordered_col_names, axis=1)
  return df

def _apply_keep_indices(df, all_split_indices, split_indices, keep_indices, sample_weights=None):
  drop_indices = np.setdiff1d(np.arange(len(df)), keep_indices)

  def update_indices(drop_indices, indices):
    # Drop any indices in the "to drop" list
    indices = np.setdiff1d(indices, drop_indices)
    # Offset the rest of them to account for deleted rows
    decrement_indices = [np.searchsorted(indices, i) for i in drop_indices]
    for dec_ind in decrement_indices:
      indices[dec_ind:] -= 1
    return indices

  for k in all_split_indices.keys():
    all_split_indices[k] = update_indices(drop_indices, all_split_indices[k])

  split_indices = update_indices(drop_indices, split_indices)

  if sample_weights is not None:
    weight_indices = sorted(sample_weights.keys())
    weight_indices_copy = np.copy(weight_indices)

    # Offset the rest of them to account for deleted rows
    decrement_indices = [np.searchsorted(weight_indices, i) for i in drop_indices]
    for dec_ind in decrement_indices:
      weight_indices_copy[dec_ind:] -= 1

    new_sample_weights = {}
    for old, new in zip(weight_indices, weight_indices_copy):
      if old not in drop_indices:
        new_sample_weights[new] = sample_weights[old]

    sample_weights = new_sample_weights

  return df.loc[keep_indices], all_split_indices, split_indices, sample_weights

def _col_val_filter(df, column_name, value_match_str):
  trues = df[column_name].str.contains(value_match_str)
  trues = trues.fillna(False)
  indices = trues[trues].index

  def checker(d, i):
    try:
      val = value_match_str in d.loc[i, column_name]
    except Exception as e:
      for i_ in d.index:
        print(i_)
      print(e)
      return False
    return val

  return indices, checker

def _inv_col_val_filter(df, column_name, value_match_str):
  indices, checker = _col_val_filter(df, column_name, value_match_str)

  inv_indices = np.setdiff1d(np.arange(len(df)), indices)
  inv_checker = lambda d, i: not checker(d, i)

  return inv_indices, inv_checker

def apply_filter(data, all_split_indices, split_indices, filter_info, sample_weights=None):
  filter_type, filter_params = filter_info
  keep_indices = None
  lens = {k: len(v) for k, v in all_split_indices.items()}
  checker = lambda d, i: True

  if filter_type == 'col_val':
    keep_indices, checker = _col_val_filter(data, **filter_params)
  elif filter_type == 'inv_col_val':
    keep_indices, checker = _inv_col_val_filter(data, **filter_params)

  if keep_indices is not None:
    data, all_split_indices, split_indices, sample_weights = _apply_keep_indices(data,
                                                                 all_split_indices,
                                                                 split_indices,
                                                                 keep_indices,
                                                                 sample_weights)
  new_sample_weights = sample_weights
  if sample_weights is not None and False:
    new_sample_weights = {}
    for k, v in sample_weights.items():
      for local_indices in all_split_indices.values():
        if k in local_indices:
          new_sample_weights[k] = v

    weights_sum = sum(new_sample_weights.values())
    for k, v in new_sample_weights.items():
      new_sample_weights[k] = v / weights_sum

  data['col_index'] = np.arange(len(data))
  data = data.reset_index(drop=True)

  if keep_indices is not None:
    for k, v in all_split_indices.items():
      if len(v) == 0:
        raise ValueError(f"Provided filter resulted in 0 items in {k} set")
      for i in v:
        if not checker(data, i):
          raise ValueError("Checker failed for i =", i)
    print(f"Filter ({filter_info}) size reduction")
    for k, v in all_split_indices.items():
      print("{}: {} -> {}".format(k, lens[k], len(v)))

    return data, all_split_indices, split_indices, new_sample_weights

  raise ValueError("Filter type: " + filter_type + " not supported")


def _read_data_individual(dataset, split, target_traits):
  data = pd.read_csv(FILE_PATHS[dataset], low_memory=False)
  # print(dataset, "file length:", len(data))
  # We may not have wave values outside 400nm and 2400nm
  # so we fill in the missing data with zeros
  data = fix_missing_values(data, 400, 2400)
  data = fix_missing_traits(data)

  all_split_indices = {}
  is_multitask = isinstance(target_traits, list)
  if is_multitask:
    for trait in target_traits:
      if trait not in constants.TRAITS:
        raise ValueError(f"The trait '{trait}' is not found")
    with np.load(MULTITASK_SPLIT_PATHS[dataset]) as splits:
      all_split_indices = {}
      for k, v in splits.items():
        all_split_indices[k] = v
  else:
    if target_traits not in constants.TRAITS:
      raise ValueError(f"The trait '{target_traits}' is not found")
    all_split_indices = load_trait_splits(target_traits, dataset)

  if split == "train":
    split_indices = all_split_indices['train']
    # Uncomment the below to reduce the training set in order to overfit
    # num_examples = 16
    # np.random.seed(4)
    # i = np.random.choice(len(split_indices), num_examples, replace=False)
    # split_indices = split_indices[i]
  elif split == "val":
    split_indices = all_split_indices['val']
  elif split == "test":
    split_indices = all_split_indices['test']
  elif split == 'all':
    split_indices = np.concatenate((all_split_indices['train'], all_split_indices['val'], all_split_indices['test']))
  else:
    raise ValueError("Need to specify train, val or test")

  return {
    'df': data,
    'all_split_indices': all_split_indices,
    'split_indices': split_indices
  }

def _weight_indices(indices, weight):
  return { i: weight for i in indices }

def _weight_all_indices(all_indices, weight):
  out = {}
  for v in all_indices.values():
    out.update(_weight_indices(v, weight))
  return out

def _weight_all_all_indices(indices, all_indices, weight):
  out = _weight_indices(indices, weight)
  out.update(_weight_all_indices(all_indices, weight))
  return out

def _merge_datasets(datasets, dataset_weights=None):
  data = datasets[0]['df']
  all_split_indices = datasets[0]['all_split_indices']
  split_indices = datasets[0]['split_indices']
  all_index_weights = None
  if dataset_weights is not None:
    all_index_weights = _weight_all_all_indices(split_indices, all_split_indices, dataset_weights[0])

  for i, dataset in enumerate(datasets[1:]):
    i_ = i + 1
    index_offset = np.max(data.index) + 1
    dataset['df']['col_index'] += index_offset
    data = pd.concat([data, dataset['df']], axis=0, ignore_index=True, sort=False)
    for k in dataset['all_split_indices'].keys():
      new_split_indices = dataset['all_split_indices'][k] + index_offset
      if dataset_weights is not None:
        all_index_weights.update(_weight_indices(new_split_indices, dataset_weights[i_]))
      all_split_indices[k] = np.concatenate([all_split_indices[k],
                                             new_split_indices],
                                            axis=0)
    new_split_indices = dataset['split_indices'] + index_offset
    if dataset_weights is not None:
      all_index_weights.update(_weight_indices(new_split_indices, dataset_weights[i_]))
    split_indices = np.concatenate([split_indices,
                                    new_split_indices],
                                    axis=0)
  for k in all_split_indices.keys():
    all_split_indices[k].sort()
  split_indices.sort()

  if dataset_weights is not None:
    weights_sum = sum(all_index_weights.values())
    for k, v in all_index_weights.items():
      all_index_weights[k] = v / weights_sum

  return data, all_split_indices, split_indices, all_index_weights


def _get_dataset_weights(datasets):
  out_weights = []
  total_example_count = 0
  for dataset in datasets:
    for k, v in dataset['all_split_indices'].items():
      total_example_count += len(v)
  for dataset in datasets:
    examples_in_dataset = sum([len(v) for v in dataset['all_split_indices'].values()])
    out_weights.append(1 / (examples_in_dataset / total_example_count))

  return np.asarray(out_weights) / np.sum(out_weights)


def read_data(dataset, split, target_traits, filter_info=None, weight_datasets=False):
  dataset_names = dataset.split('+')
  individ_datasets = [_read_data_individual(d, split, target_traits) for d in dataset_names]

  dataset_weights = _get_dataset_weights(individ_datasets) if weight_datasets else None

  data, all_split_indices, split_indices, sample_weights = _merge_datasets(individ_datasets, dataset_weights)

  if filter_info is not None:
    data, all_split_indices, split_indices, sample_weights = apply_filter(data,
                                                                          all_split_indices,
                                                                          split_indices,
                                                                          filter_info,
                                                                          sample_weights)
  combined_weights = sample_weights
  # This - when enabled - kinda balances the traits by sampling rows with under-represented
  # traits more often. Experiments show this doesn't help
  if weight_datasets and False:
    trait_weights = _weight_trait_representation(data, all_split_indices, target_traits)
    combined_weights = _combine_trait_sample_weights(trait_weights, sample_weights)

  return data, all_split_indices, split_indices, combined_weights


def _weight_trait_representation(df, all_split_indices, traits):
  from collections import defaultdict
  total_count = sum(len(v) for v in all_split_indices.values())
  trait_counts = defaultdict(lambda: 0)
  for trait in traits:
    for k, v in all_split_indices.items():
      trait_counts[trait] += df.loc[v, trait].count()

  trait_percentages = {
    k: v / total_count for k, v in trait_counts.items()
  }

  index_weights = {}
  for k, v in all_split_indices.items():
    for i in v:
      local_weights = []
      for trait in traits:
        if not pd.isna(df.loc[i, trait]):
          local_weights.append(trait_percentages[trait])
      index_weights[i] = 1 /np.prod(local_weights)
  weights_sum = sum(index_weights.values())
  index_weights = {
    k: v / weights_sum for k, v in index_weights.items()
  }
  return index_weights

def _combine_trait_sample_weights(trait_weights, sample_weights):
  combined_weights = {}
  for k in trait_weights.keys():
    combined_weights[k] = trait_weights[k] * sample_weights[k]
  combined_sum = sum(combined_weights.values())
  return {k: v / combined_sum for k, v in combined_weights.items()}
