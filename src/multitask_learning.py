from time import time
import copy

import numpy as np
import pandas as pd
import torch
from terminalplot import plot

from dataset import HyperSpectrumMultiDataSet as hsmd


def clean_dict_nan_vals(d):
  cleaning = True
  while cleaning:
    cleaning = False
    for k, v in d.items():
      if np.isnan(v):
        del d[k]
        cleaning = True
        break


def load_ds(c, traits,
            dataset="train",
            data_dir="/data",
            batch_size=16,
            include_wavelengths=False,
            augmentation=None,
            variable_length_params=None,
            dataset_version='v1',
            oversample=False,
            shuffle=True):
    ds = hsmd.HyperSpectrumMultiDataSet(c, traits,
                                        dataset,
                                        data_dir,
                                        include_wavelengths,
                                        variable_length_params,
                                        dataset_version,
                                        oversample=oversample)
    if augmentation:
      for aug in augmentation:
          [aug_type, aug_percent, rate] = aug
          ds.data_augmentation(aug_type, aug_percent, rate)

    oversampler = None
    if oversample:
      print("<< Balancing with OverSampler >>")
      shuffle = False
      oversampler = ds.get_oversampler()
    ds_loader = torch.utils.data.DataLoader(ds,
                                            batch_size=batch_size,
                                            shuffle=shuffle,
                                            collate_fn=ds.collate_fn,
                                            sampler=oversampler)

    return ds, ds_loader


def filter_nan_labels(preds, labels, return_indices=False):
  """Drop predictions and labels where the label is nan (missing vals)"""
  nan_mask = torch.isnan(labels)
  unmasked_indices = torch.nonzero(nan_mask == 0)[:, 0]
  labels = labels[unmasked_indices]
  preds = preds[unmasked_indices]

  if return_indices:
    return preds, labels, unmasked_indices
  return preds, labels


def compute_label_sq_diffs(traits, dataset):
    non_nan_trait_vals = {}
    label_sq_diffs = {}

    for trait in traits:
        non_nan_trait_vals[trait] = []

    for i in range(len(dataset)):
        data = dataset[i]

        for idx, trait in enumerate(traits):
            if not np.isnan(data['label'][idx]):
                non_nan_trait_vals[trait].append(data['label'][idx])

    label_means = {}
    for trait in traits:
        non_nan_trait_vals[trait] = np.array(non_nan_trait_vals[trait])
        label_means[trait] = np.mean(non_nan_trait_vals[trait])

    for trait in traits:
      label_sq_diffs[trait] = np.sum(np.square((non_nan_trait_vals[trait] - label_means[trait])))
    return label_sq_diffs


def compute_r_square(model, traits, dataset, dsloader):
    label_sq_diffs = compute_label_sq_diffs(traits, dataset)
    pred_sq_diffs = {}
    r_squares = {}
    for trait in traits:
        pred_sq_diffs[trait] = 0
    model.eval()
    for i, data in enumerate(dsloader, 0):
        values = data["value"].to(model.device)

        traits_outputs = model(values)  # produce multiple label

        for idx, trait in enumerate(traits):
            labels = data['label'][:, idx]
            labels = labels.unsqueeze(1)

            outputs = traits_outputs[:, idx].unsqueeze(1)

            outputs, labels = filter_nan_labels(outputs, labels)
            outputs = outputs.cpu()
            # compute r square
            diff = outputs - labels

            pred_sq_diffs[trait] += torch.sum(torch.mul(diff, diff), 0).detach().numpy()

    for idx, trait in enumerate(traits):
        if np.isfinite(label_sq_diffs[trait]) and label_sq_diffs[trait] != 0:
          r_squares[trait] = 1 - float(pred_sq_diffs[trait]) / float(label_sq_diffs[trait])
        else:
          r_squares[trait] = np.nan

    avg_r_square = np.mean([v for v in r_squares.values() if np.isfinite(v)])

    return avg_r_square, r_squares


def predict(model, traits, dataset, dsloader):
    predicted_labels = {}
    for trait in traits:
        predicted_labels[trait] = pd.DataFrame(columns=['observed', 'predicted'])
    for i, data in enumerate(dsloader, 0):
        values = data["value"].to(model.device)
        traits_outputs = model(values)  # produce multiple label
        for idx, trait in enumerate(traits):
            labels = data['label'][:, idx]
            labels = labels.unsqueeze(1)
            outputs = traits_outputs[:, idx].unsqueeze(1)
            outputs, labels = filter_nan_labels(outputs, labels)

            # denormalize for calculate bias and rep
            labels = dataset.denormalize(trait, labels)
            outputs = dataset.denormalize(trait, outputs)

            labels_df = pd.DataFrame(labels.detach().cpu().numpy())
            labels_df.rename(columns={0: 'observed'}, inplace=True)
            outputs_df = pd.DataFrame(outputs.detach().cpu().numpy())
            outputs_df.rename(columns={0: 'predicted'}, inplace=True)

            predicted_labels[trait] = predicted_labels[trait].append(pd.concat([labels_df, outputs_df], axis=1), ignore_index=True)

    return predicted_labels


def compute_results(model, traits, dataset, dsloader, with_additional=False):
    label_sq_diffs = compute_label_sq_diffs(traits, dataset)
    pred_sq_diffs = {}
    d_pred_sq_diffs = {} #denormalized
    mean_outputs = {}
    mean_labels = {}
    num_data = {}
    sum_outputs = {}
    sum_labels = {}
    r_squares = {}
    bias_values = {}
    rep_values = {}
    for trait in traits:
        pred_sq_diffs[trait] = 0
        d_pred_sq_diffs[trait] = 0
        mean_outputs[trait] = 0
        mean_labels[trait] = 0
        num_data[trait] = 0
        sum_outputs[trait] = 0
        sum_labels[trait] = 0

    all_outputs = {}
    for t in traits:
      all_outputs[t] = {
        'predictions': [],
        'labels': [],
        'indices': [],
        'l1_errors': []
      }
      if with_additional:
        all_outputs[t]['additional'] = {}

    model.eval()
    index_offset = 0
    for i, data in enumerate(dsloader, 0):
        values = data["value"].to(model.device)

        traits_outputs = model(values)  # produce multiple label

        for idx, trait in enumerate(traits):
            labels = data['label'][:, idx]
            labels = labels.unsqueeze(1)
            outputs = traits_outputs[:, idx].unsqueeze(1)
            outputs, labels, indices = filter_nan_labels(outputs, labels, return_indices=True)
            outputs = outputs.cpu()
            # compute r square
            diff = outputs - labels
            pred_sq_diffs[trait] += torch.sum(torch.mul(diff, diff), 0).detach().numpy()

            # denormalize for calculate bias and rep
            labels = dataset.denormalize(trait, labels)
            outputs = dataset.denormalize(trait, outputs)

            all_outputs[trait]['labels'].extend(labels.detach().numpy())
            all_outputs[trait]['indices'].extend([dataset.split_indices[i + index_offset] for i in indices.detach().numpy()])
            all_outputs[trait]['predictions'].extend(outputs.detach().numpy())
            all_outputs[trait]['l1_errors'].extend(np.abs(diff.detach().numpy()))

            if with_additional:
              additional = model.additional_outputs()
              if additional is not None:
                for k, v in additional.items():
                  if k not in all_outputs[trait]['additional']:
                    all_outputs[trait]['additional'][k] = []
                  all_outputs[trait]['additional'][k].extend(v)

            d_diff = outputs - labels
            d_pred_sq_diffs[trait] += torch.sum(torch.mul(d_diff, d_diff), 0).detach().numpy()

            sum_labels[trait] += np.sum(labels.detach().numpy())
            sum_outputs[trait] += np.sum(outputs.detach().numpy())
            num_data[trait] += len(labels)
        index_offset += len(values)

    for trait in traits:
        if num_data[trait] > 0:
          r_squares[trait] = 1 - pred_sq_diffs[trait] / label_sq_diffs[trait]
          mean_labels[trait] = (sum_labels[trait] / num_data[trait])
          mean_outputs[trait] = (sum_outputs[trait] / num_data[trait])
          bias_values[trait] = 100.0 * (mean_outputs[trait] - mean_labels[trait]) / mean_labels[trait]
          rep_values[trait] = 100.0 * np.sqrt(d_pred_sq_diffs[trait] / num_data[trait]) / mean_labels[trait]

          lbls = np.asarray(all_outputs[trait]['labels']).reshape(-1)
          preds = np.asarray(all_outputs[trait]['predictions']).reshape(-1)
          all_outputs[trait]['correlation'] = np.corrcoef(lbls, preds)[0, -1]
        else:
          all_outputs[trait]['correlation'] = np.nan

    avg_r_square = np.mean([v for v in r_squares.values() if np.isfinite(v)])
    return avg_r_square, r_squares, bias_values, rep_values, all_outputs


def train_multitask_learning_v2(c, model, criterion, optimizer, traits, data_dir,
                                batch_size=16, num_epochs=1000,
                                include_wavelengths=False, writer=None, variable_length_params=None):
    test_every = 10

    train_ds, trainloader = load_ds(c,
                                    traits,
                                    "train",
                                    data_dir,
                                    batch_size,
                                    augmentation=c.augmentation,
                                    include_wavelengths=include_wavelengths,
                                    variable_length_params=variable_length_params,
                                    dataset_version=c.dataset)
    val_ds, validationloader = load_ds(c,
                                       traits,
                                       "val",
                                       data_dir,
                                       batch_size,
                                       include_wavelengths=include_wavelengths,
                                       variable_length_params=variable_length_params,
                                       dataset_version=c.dataset)

    if isinstance(traits, str):
      traits = [traits]

    train_label_sq_diffs = compute_label_sq_diffs(traits, train_ds)

    best_overall_model = copy.deepcopy(model)
    best_avg_val_R_square = -99999.99

    train_results = {}
    best_models = {}
    best_val_R_squares = {}
    best_val_bias = {}
    best_val_REP = {}
    best_model_epoch = 0

    for trait in traits:
        best_models[trait] = copy.deepcopy(model)
        best_val_R_squares[trait] = -99999.99
        best_val_bias[trait] = -99999.99
        best_val_REP[trait] = -99999.99

    val_r_squares = []
    time_per_example_ma = 0
    MA = 0.5
    examples_per_epoch = len(train_ds)
    print("Train set size:", examples_per_epoch)
    for epoch in range(num_epochs):
        print("\rEpoch {}/{}".format(epoch, num_epochs), end='')
        running_loss = 0.0
        pred_sq_diffs = {}
        train_R_squares = {}

        for trait in traits:
            pred_sq_diffs[trait] = 0
        model.train()
        epoch_start_time = time()
        for i, data in enumerate(trainloader, 0):
            values = data["value"].to(model.device)

            traits_labels = data["label"]

            # zero the parameter gradients
            optimizer.zero_grad()
            traits_outputs = model(values)  # produce multiple label
            traits_outputs = traits_outputs.cpu()

            # forward + backward + optimize
            loss = 0
            for idx, trait in enumerate(traits):
                labels = traits_labels[:, idx].unsqueeze(1)
                outputs = traits_outputs[:, idx].unsqueeze(1)
                outputs, labels = filter_nan_labels(outputs, labels)
                # sum all loss for each trait
                assert len(outputs) == len(labels)
                if len(outputs) > 0:
                  loss += criterion(outputs, labels)

                  # compute r square
                  diff = outputs - labels
                  pred_sq_diffs[trait] += torch.sum(torch.mul(diff, diff), 0).detach().numpy()

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
        epoch_end_time = time()
        time_per_example = (epoch_end_time - epoch_start_time) / examples_per_epoch
        time_per_example_ma = MA * time_per_example_ma + (1 - MA) * time_per_example

        # compute r scores for each trait
        for trait in traits:
            train_R_squares[trait] = 1 - pred_sq_diffs[trait] / train_label_sq_diffs[trait]

        clean_dict_nan_vals(train_R_squares)
        avg_train_R_square = np.mean(list(train_R_squares.values()))
        if epoch % test_every == 0:
            #avg_val_R_square, val_R_squares = compute_r_square(model, traits, val_ds, validationloader)
            avg_val_R_square, val_R_squares, val_bias_values, val_rep_values, _ = compute_results(model, traits, val_ds, validationloader)

            print("\nEpoch: {:d}; time per example: {:.3f}sec; examples per second: {:d}"
                  .format(epoch, time_per_example_ma, int(round(1 / time_per_example_ma))))

            for trait in traits:
              if trait in train_R_squares.keys():
                  if np.isfinite(train_R_squares[trait]):
                    print("--Trait: {:s}\n\tTrain R square. {:.4f}.\tVal R square: {:.4f}".format(trait, np.asscalar(
                                                        train_R_squares[trait]), np.asscalar(val_R_squares[trait])))
            print(" Avg train R square: {:.4f}.\tAvg Val R square: {:.4f}".format(np.asscalar(avg_train_R_square),
                                                                                 np.asscalar(avg_val_R_square)))
            val_r_squares.append(np.asscalar(avg_val_R_square))

            plot(
              (np.arange(len(val_r_squares)) * test_every).tolist(),
              val_r_squares,
              rows=15,
              columns=40
            )
            clean_dict_nan_vals(val_R_squares)
            writer.add_scalars(
              'rSquare/train',
              train_R_squares,
              epoch
            )
            writer.add_scalars(
              'rSquare/val',
              val_R_squares,
              epoch
            )
            writer.add_scalar('rSquare/mean/train', avg_train_R_square, epoch)
            writer.add_scalar('rSquare/mean/val', avg_val_R_square, epoch)


            for trait in traits:
                if trait in val_R_squares.keys() and best_val_R_squares[trait] < val_R_squares[trait]:
                    best_models[trait] = copy.deepcopy(model)
                    best_val_R_squares[trait] = val_R_squares[trait]
                    best_val_bias[trait] = val_bias_values[trait]
                    best_val_REP[trait] = val_rep_values[trait]
                    train_results[trait] = train_R_squares[trait]


            if best_avg_val_R_square < avg_val_R_square:
                best_overall_model = copy.deepcopy(model)
                best_avg_val_R_square = avg_val_R_square
                best_model_epoch = epoch

    print("Best avg validation_rsquare: {:4f} ".format(np.asscalar(best_avg_val_R_square)))
    print("  Found at epoch", best_model_epoch)
    return best_overall_model, best_models, train_results, best_val_R_squares, best_val_bias, best_val_REP
