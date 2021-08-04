import unittest

import pandas as pd
import numpy as np

import dataset.utils as utils
import dataset.constants as constants
from dataset.HyperSpectrumMultiDataSet import OverSampler


class DatasetTestCase(unittest.TestCase):
  def test_merge(self):
    """Merging results in the same train/val/test rows"""
    train_rows = []
    val_rows = []
    test_rows = []

    datasets = []
    df = pd.DataFrame([['b', 2, 'banana'],
                       ['a', 1, 'apple'],
                       ['d', 4, 'durian'],
                       ['c', 3, 'cherry']],
                      columns=['letter', 'number', 'fruit'])
    df['col_index'] = range(0, len(df))
    all_split_indices = {
      'train': np.array([2, 0]),
      'val': np.array([1]),
      'test': np.array([3])
    }
    datasets.append({
      'df': df,
      'all_split_indices': all_split_indices,
      'split_indices': all_split_indices['train']
    })
    for i in all_split_indices['train']:
      train_rows.append(df.loc[i]['letter'])
    for i in all_split_indices['val']:
      val_rows.append(df.loc[i]['letter'])
    for i in all_split_indices['test']:
      test_rows.append(df.loc[i]['letter'])

    df = pd.DataFrame([['f', np.nan, 'fish'],
                       ['g', 7, 'goat'],
                       ['e', 5, 'elephant'],
                       ['i', 9, 'insect'],
                       ['h', 8, 'horse']],
                      columns=['letter', 'number', 'thing'])
    df['col_index'] = range(0, len(df))
    all_split_indices = {
      'train': np.array([0, 4]),
      'val': np.array([3, 1]),
      'test': np.array([2])
    }
    datasets.append({
      'df': df,
      'all_split_indices': all_split_indices,
      'split_indices': all_split_indices['train']
    })
    for i in all_split_indices['train']:
      train_rows.append(df.loc[i]['letter'])
    for i in all_split_indices['val']:
      val_rows.append(df.loc[i]['letter'])
    for i in all_split_indices['test']:
      test_rows.append(df.loc[i]['letter'])

    data, all_split_indices, split_indices, _ = utils._merge_datasets(datasets)

    # Test data is the right length
    np.testing.assert_equal(len(data), sum([len(d['df']) for d in datasets]))

    # Test that the same rows are in each split
    for i in all_split_indices['train']:
      assert data.loc[i]['letter'] in train_rows
    for i in all_split_indices['val']:
      assert data.loc[i]['letter'] in val_rows
    for i in all_split_indices['test']:
      assert data.loc[i]['letter'] in test_rows

  def test_merge2(self):
    """Merging 3 datasets results in the same train/val/test rows"""
    train_rows = []
    val_rows = []
    test_rows = []

    datasets = []
    df = pd.DataFrame([['b', 2, 'banana'],
                       ['a', 1, 'apple'],
                       ['d', 4, 'durian'],
                       ['c', 3, 'cherry']],
                      columns=['letter', 'number', 'thing'])
    df['col_index'] = range(0, len(df))
    all_split_indices = {
      'train': np.array([2, 0]),
      'val': np.array([1]),
      'test': np.array([3])
    }
    datasets.append({
      'df': df,
      'all_split_indices': all_split_indices,
      'split_indices': all_split_indices['train']
    })
    for i in all_split_indices['train']:
      train_rows.append(df.loc[i]['letter'])
    for i in all_split_indices['val']:
      val_rows.append(df.loc[i]['letter'])
    for i in all_split_indices['test']:
      test_rows.append(df.loc[i]['letter'])

    df = pd.DataFrame([['f', np.nan, 'fish'],
                       ['g', 7, 'goat'],
                       ['e', 5, 'elephant'],
                       ['i', 9, 'insect'],
                       ['h', 8, 'horse']],
                      columns=['letter', 'number', 'thing'])
    df['col_index'] = range(0, len(df))
    all_split_indices = {
      'train': np.array([0, 4]),
      'val'  : np.array([3, 1]),
      'test' : np.array([2])
    }
    datasets.append({
      'df'               : df,
      'all_split_indices': all_split_indices,
      'split_indices'    : all_split_indices['train']
    })
    for i in all_split_indices['train']:
      train_rows.append(df.loc[i]['letter'])
    for i in all_split_indices['val']:
      val_rows.append(df.loc[i]['letter'])
    for i in all_split_indices['test']:
      test_rows.append(df.loc[i]['letter'])

    df = pd.DataFrame([['k', 21, 'house'],
                       ['j', 20, 'car'],
                       ['l', 22, 'boat']],
                      columns=['letter', 'number', 'thing'])
    df['col_index'] = range(0, len(df))
    all_split_indices = {
      'train': np.array([0]),
      'val'  : np.array([2]),
      'test' : np.array([1])
    }
    datasets.append({
      'df'               : df,
      'all_split_indices': all_split_indices,
      'split_indices'    : all_split_indices['train']
    })
    for i in all_split_indices['train']:
      train_rows.append(df.loc[i]['letter'])
    for i in all_split_indices['val']:
      val_rows.append(df.loc[i]['letter'])
    for i in all_split_indices['test']:
      test_rows.append(df.loc[i]['letter'])


    weights = utils._get_dataset_weights(datasets)
    data, all_split_indices, split_indices, sample_weights = utils._merge_datasets(datasets, weights)

    # Test data is the right length
    np.testing.assert_equal(len(data), sum([len(d['df']) for d in datasets]))

    # Test that the same rows are in each split
    for i in all_split_indices['train']:
      assert data.loc[i]['letter'] in train_rows
    for i in all_split_indices['val']:
      assert data.loc[i]['letter'] in val_rows
    for i in all_split_indices['test']:
      assert data.loc[i]['letter'] in test_rows

  def test_weighting(self):
    """Test that three unbalanced datasets are weighed correctly"""
    train_rows = []
    val_rows = []
    test_rows = []

    datasets = []
    df = pd.DataFrame([['b', 2, 'banana'],
                       ['a', 1, 'apple'],
                       ['d', 4, 'durian'],
                       ['c', 3, 'cherry']],
                      columns=['letter', 'number', 'thing'])
    df['col_index'] = range(0, len(df))
    all_split_indices = {
      'train': np.array([2, 0]),
      'val'  : np.array([1]),
      'test' : np.array([3])
    }
    datasets.append({
      'df'               : df,
      'all_split_indices': all_split_indices,
      'split_indices'    : all_split_indices['train']
    })
    for i in all_split_indices['train']:
      train_rows.append(df.loc[i]['letter'])
    for i in all_split_indices['val']:
      val_rows.append(df.loc[i]['letter'])
    for i in all_split_indices['test']:
      test_rows.append(df.loc[i]['letter'])

    df = pd.DataFrame([['f', np.nan, 'fish'],
                       ['g', 7, 'goat'],
                       ['e', 5, 'elephant'],
                       ['i', 9, 'insect'],
                       ['h', 8, 'horse']],
                      columns=['letter', 'number', 'thing'])
    df['col_index'] = range(0, len(df))
    all_split_indices = {
      'train': np.array([0, 4]),
      'val'  : np.array([3, 1]),
      'test' : np.array([2])
    }
    datasets.append({
      'df'               : df,
      'all_split_indices': all_split_indices,
      'split_indices'    : all_split_indices['train']
    })
    for i in all_split_indices['train']:
      train_rows.append(df.loc[i]['letter'])
    for i in all_split_indices['val']:
      val_rows.append(df.loc[i]['letter'])
    for i in all_split_indices['test']:
      test_rows.append(df.loc[i]['letter'])

    df = pd.DataFrame([['k', 21, 'house'],
                       ['j', 20, 'car'],
                       ['l', 22, 'boat']],
                      columns=['letter', 'number', 'thing'])
    df['col_index'] = range(0, len(df))
    all_split_indices = {
      'train': np.array([0]),
      'val'  : np.array([2]),
      'test' : np.array([1])
    }
    datasets.append({
      'df'               : df,
      'all_split_indices': all_split_indices,
      'split_indices'    : all_split_indices['train']
    })
    for i in all_split_indices['train']:
      train_rows.append(df.loc[i]['letter'])
    for i in all_split_indices['val']:
      val_rows.append(df.loc[i]['letter'])
    for i in all_split_indices['test']:
      test_rows.append(df.loc[i]['letter'])


    weights = utils._get_dataset_weights(datasets)
    np.testing.assert_equal(sum(weights), 1)
    np.testing.assert_equal(len(weights), len(datasets))

    data, all_split_indices, split_indices, sample_weights = utils._merge_datasets(datasets, weights)
    np.testing.assert_equal(sum(sample_weights.values()), 1)

  def test_oversampler(self):
    np.random.seed(42)
    # First 10 from one dataset, last 5 from another. Thus the last 5 are weighted twice as high
    count = 15
    double_weighted_index = 10

    sample_weights = np.ones(count)
    sample_weights[double_weighted_index:] = 2

    sample_weights /= np.sum(sample_weights)

    os = OverSampler(sample_weights)

    sampled = {i: 0 for i in range(count)}
    num_samples = 10000
    for _ in range(num_samples):
      for i in os.__iter__():
        sampled[i] += 1
    sampled_indices = sorted(sampled.keys())
    sampled = [sampled[i] for i in sampled_indices]
    singly_weighted = sampled[:double_weighted_index]
    doubly_weighted = sampled[double_weighted_index:]
    # Allow for a 5% discrepancy due to it being randomly sampled (this could be made stricter as
    # we draw more samples)
    tolerance = num_samples * 0.05
    np.testing.assert_array_less(abs(sum(singly_weighted) - sum(doubly_weighted)), tolerance)
