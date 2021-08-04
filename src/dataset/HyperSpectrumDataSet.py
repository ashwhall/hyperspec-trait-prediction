import json

import numpy as np
from torch.utils.data import Dataset

import dataset.utils as utils
import dataset.constants as constants


LOAD_STORED_STATISTICS = True


class HyperSpectrumDataSet(Dataset):
    """HyperSpectrum dataset."""

    def __init__(self, c, trait, split, data_dir, dataset='v1'):
        """
        Args:
            trait: "LMA_O", "Narea_O", "SPAD_O", "Nmass_O", "Parea_O", "Pmass_O",
                    "Vcmax", "Vcmax25", "J", "Photo_O", "Cond_O", "Vcmax25_Narea_O"
            split: "train", "val", "test"
        """
        data, all_split_indices, split_indices, _ = utils.read_data(dataset,
                                                                    split,
                                                                    trait,
                                                                    c.data_filter)

        train_input_df = data.loc[all_split_indices['train'], 'Wave_400':'Wave_2400']

        statistics = None
        if LOAD_STORED_STATISTICS:
          with open(constants.STAT_DUMP_PATH, 'r') as f:
            statistics = json.load(f)

        #normalize input
        if statistics is not None:
          value_average = statistics['values']['mean']
          value_std = statistics['values']['std']
        else:
          value_average = train_input_df.values.mean().astype(np.float32)
          value_average = np.asscalar(value_average)
          value_std = train_input_df.values.std().astype(np.float32)
          value_std = np.asscalar(value_std)


        # This is the default range
        self.data_values = data.loc[split_indices, 'Wave_400':'Wave_2400'] - value_average

        # These three are used for training the different PLS ranges. Disable and enable as required
        # self.data_values = data.loc[split_indices, 'Wave_400':'Wave_900'] - value_average
        # self.data_values = data.loc[split_indices, 'Wave_400':'Wave_1000'] - value_average
        # self.data_values = data.loc[split_indices, 'Wave_400':'Wave_1700'] - value_average

        self.data_values = self.data_values / value_std

        print(len(self.data_values), "examples for", split)

        print(data.loc[split_indices, trait].size, '->', data.loc[split_indices, trait].count())
        for i in split_indices:
          assert np.isfinite(data.loc[i, trait]), f"Non finite value found for trait {trait}"

        #normalize target trait
        train_trait_df = data.loc[all_split_indices['train'], [trait]]
        self.trait_average = train_trait_df.mean().values.astype(np.float32)
        self.trait_average = np.asscalar(self.trait_average)
        self.trait_std = train_trait_df.std().values.astype(np.float32)
        self.trait_std = np.asscalar(self.trait_std)

        self.data_labels = data.loc[split_indices,[trait]] - self.trait_average
        self.data_labels = self.data_labels / self.trait_std


        labels_numpy = np.squeeze(self.data_labels.values)
        mean = np.mean(labels_numpy)
        self.sq_diff = np.sum(np.square(labels_numpy - mean))
        self.split_indices = split_indices

    def __len__(self):
        return len(self.data_values)

    def get_sq_diff(self):
        return self.sq_diff

    # denormalize labels
    def denormalize(self, labels):
        return labels * self.trait_std + self.trait_average

    def __getitem__(self, idx):
        item = {}
        item["value"] = self.data_values.iloc[idx,:].values.astype(np.float32)
        item["label"] = self.data_labels.iloc[idx,0].astype(np.float32)
        return item
