import json
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.stats import truncnorm

import dataset.utils as utils
import dataset.constants as constants


LOAD_STORED_STATISTICS = True
STORE_STATISTICS = False
assert not (LOAD_STORED_STATISTICS and STORE_STATISTICS), "Pretty useless to load and store statistics right away!"


class HyperSpectrumMultiDataSet(Dataset):
    """HyperSpectrum multi-task dataset."""
    def __init__(self, c, target_traits, split, data_dir, include_wavelengths, variable_length_params=None, dataset_version='v1', oversample=False):
      """
      Args:
          trait: "LMA_O", "Narea_O", "SPAD_O", "Nmass_O", "Parea_O", "Pmass_O",
                  "Vcmax", "Vcmax25", "J", "Photo_O", "Cond_O"
          split: "train", "val", "test"
      """
      self.include_wavelengths = include_wavelengths
      self.variable_length_params = variable_length_params
      self.sampled = defaultdict(lambda: 0)
      if isinstance(target_traits, str):
        traits = [target_traits]
      else:
        traits = target_traits

      data, all_split_indices, split_indices, sample_weights = utils.read_data(dataset_version,
                                                                               split,
                                                                               target_traits,
                                                                               c.data_filter,
                                                                               weight_datasets=oversample)

      self._sample_weights = sample_weights
      if self._sample_weights is not None:
        self._sample_weights = np.array([sample_weights[i] for i in split_indices])

      # normalize input
      train_input_df = data.loc[all_split_indices['train'], 'Wave_400':'Wave_2400']

      statistics = None
      if LOAD_STORED_STATISTICS:
        with open(constants.STAT_DUMP_PATH, 'r') as f:
          statistics = json.load(f)

      if statistics is not None:
        value_average = statistics['values']['mean']
        value_std = statistics['values']['std']
      else:
        value_average = train_input_df.values.mean().astype(np.float32)
        value_average = np.asscalar(value_average)
        value_std = train_input_df.values.std().astype(np.float32)
        value_std = np.asscalar(value_std)

      self.data_values = data.loc[split_indices, 'Wave_400':'Wave_2400'] - value_average
      self.data_values = self.data_values / value_std

      # Change wavelengths so they're a linspace in [0.4, 2.4]
      self.wavelengths = np.array([int(c[c.rindex('_') + 1:]) for c in self.data_values.columns], np.float32)
      self.wavelengths = self.wavelengths / 1000


      self.trait_averages = {}
      self.trait_stds = {}
      for trait in traits:
        # normalize traits
        train_df = data.loc[all_split_indices['train'], [trait]]
        train_df = train_df.dropna(axis=0, how='any')

        # Uncomment the below to normalise traits using the loaded statistics
        # if statistics is not None:
        #   trait_average = statistics['traits']['mean'][trait]
        #   trait_std = statistics['traits']['std'][trait]
        # else:
        trait_average = train_df.mean().values.astype(np.float32)
        trait_average = np.asscalar(trait_average)
        trait_std = train_df.std().values.astype(np.float32)
        trait_std = np.asscalar(trait_std)

        data.loc[split_indices, [trait]] = data.loc[split_indices, [trait]] - trait_average
        data.loc[split_indices, [trait]] = data.loc[split_indices, [trait]] / trait_std
        self.trait_averages[trait] = trait_average
        self.trait_stds[trait] = trait_std

      if STORE_STATISTICS:
        statistics = {
          'values': {
            'mean': value_average,
            'std' : value_std
          },
          'traits': {
            'mean': self.trait_averages,
            'std' : self.trait_stds
          }
        }
        # print(statistics)
        with open(constants.STAT_DUMP_PATH, 'w') as f:
          json.dump(statistics, f)

      self.data_labels = data.loc[split_indices, traits]

      labels_numpy = np.squeeze(self.data_labels.values)
      mean = np.mean(labels_numpy)
      self.sq_diff = np.sum(np.square(labels_numpy - mean))
      self.split_indices = split_indices

    def get_oversampler(self):
      return OverSampler(self._sample_weights)

    # data augmentation
    def data_augmentation(self, aug_type="shift_updown", aug_percent=0.1, rate=0):
        num_data = len(self.data_labels.values)
        num_added = int(num_data * aug_percent)
        print("num_added: ", num_added)

        randNum = np.random.randint(num_data, size=num_added)
        new_labels = []
        new_values = []
        include_weights = self._sample_weights is not None
        if include_weights:
          new_weights = []
        for idx in randNum:
            value = self.data_values.iloc[idx]
            label = self.data_labels.iloc[idx]
            if include_weights:
              weight = self._sample_weights[idx]

            if aug_type == "shift_updown":
                random_rate = np.random.uniform(-rate, rate)
                # random_rate = self.trunc_norm(0, rate)
                new_value = value * (1 + random_rate)
            elif aug_type == "shift_leftright":
                random_rate = np.random.randint(-rate, rate, size=1)
                # random_rate = self.trunc_norm(0, rate, np.int32)
                new_value = pd.Series(np.roll(value.values, random_rate), index=value.index)
            elif aug_type == 'linear_skew':
                start_skew = np.random.uniform(-rate, rate)
                end_skew = np.random.uniform(-rate, rate)
                # start_skew = self.trunc_norm(0, rate)
                # end_skew = self.trunc_norm(0, rate)
                skew_vals = np.linspace(1 + start_skew, 1 + end_skew, len(value))
                new_value = value * skew_vals
            else:
              raise ValueError(f"Unsupported augmentation method '{aug_type}'")

            new_labels.append(label)
            new_values.append(new_value)
            if include_weights:
              new_weights.append(weight)

        self.data_values = pd.DataFrame.append(self.data_values, new_values)
        self.data_labels = pd.DataFrame.append(self.data_labels, new_labels)
        if include_weights:
          self._sample_weights = np.concatenate((self._sample_weights, new_weights))

    def get_sq_diff(self):
        return self.sq_diff

    def __len__(self):
        return len(self.data_values)

    def __getitem__(self, idx):
        self.sampled[idx] += 1
        item = {}
        item["value"] = self.data_values.iloc[idx, :].values.astype(np.float32)
        if self.include_wavelengths:
            item["value"] = np.vstack([item["value"], self.wavelengths])

        # item["value"] = response
        item["label"] = self.data_labels.iloc[idx, :].values.astype(np.float32)

        return item

    @staticmethod
    def trunc_norm(centre, width, dtype=np.float32):
      val = truncnorm(-1, 1, loc=centre, scale=width).rvs()
      if np.issubdtype(dtype, np.integer):
        val = np.round(val)
      return val.astype(dtype)

    @staticmethod
    def rand_half_normal(centre, limit, std=0.5):
      """
      Produces a normal distribution with mean=centre, where all values below, and
      on the opposite side of limit are discarded. Thus it looks like:
                           ****
                        *******
                   ************
            *******************
      -----L------------------C----------
      where L and C are limit and centre, respectively.
      Works with limit on either side of the centre.
      """
      if limit < centre:
        lower = limit
        upper = centre
      else:
        lower = centre
        upper = limit

      return truncnorm(
        (lower - centre) / std, (upper - centre) / std, loc=centre, scale=std).rvs()

    def apply_variable_length_zeroing(self, vals, params):
        if params.get('random', True):
          lower = self.rand_half_normal(self.wavelengths[0],
                                        params['max_lower'],
                                        params['lower_std'])
          upper = self.rand_half_normal(self.wavelengths[-1],
                                        np.maximum(params['min_upper'], lower + params['min_width']),
                                        params['upper_std'])
        else:
          lower = params['max_lower']
          upper = params['min_upper']
        assert lower < upper, "lower should never be greater than upper!"

        mask = np.logical_or(self.wavelengths < lower, self.wavelengths > upper)
        if vals.ndim == 2:
          vals[:, mask] = 0
        else:
          vals[mask] = 0

    def denormalize(self, trait, labels):
        return labels * self.trait_stds[trait] + self.trait_averages[trait]

    def collate_fn(self, batch):
      """Gather batches, and zero-out the ends to simulate variable-length inputs"""
      value = np.stack([b['value'] for b in batch], 0)
      label = np.stack([b['label'] for b in batch], 0)
      if self.variable_length_params:
        for i in range(value.shape[0]):
          self.apply_variable_length_zeroing(value[i], self.variable_length_params)

      return {
        'value': torch.tensor(value, dtype=torch.float32),
        'label': torch.tensor(label, dtype=torch.float32)
      }


class OverSampler(torch.utils.data.sampler.Sampler):
  """
  Oversamples from under-represented datasets.
  split_indices is the indices into the entire DataFrame for this sampler's partition.
  sample_weights is a dict mapping indices to weights
  batch_size is self-explanatory.
  The __iter__ function returns an index in the range [0, len(split_indices)), using
  weighted sampling
  """
  def __init__(self, sample_weights):
    self._sample_weights = sample_weights / sum(sample_weights)
    self.len = len(self._sample_weights)

    # If all the weights match, we don't need to oversample
    self._all_weights_match = np.all(self._sample_weights == np.mean(self._sample_weights))

  def __iter__(self):
    # If all weights were identical, we don't need to weight things differently and can sample
    # uniformly without replacement - thus ensuring that every example is seen once per epoch
    if self._all_weights_match:
      indices = np.random.choice(self.len, size=self.len, replace=False)
    else:
      indices = np.random.choice(self.len, size=self.len, p=self._sample_weights)
    return iter(indices)

  def __len__(self):
    return self.len

