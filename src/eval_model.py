import os
from collections import defaultdict
import pickle
from time import time
import pandas as pd
import torch
import numpy as np

import config
import multitask_learning as multitask
from models.utils import load_model


def convert_df(data_dict, column_name):
    df = pd.DataFrame.from_dict(data_dict, orient='index')
    df = df.rename(index=str, columns={0: column_name})
    return df

def write_out_results(result_file, results):
    total = 0.0
    for key, value in results.items():
        if isinstance(value, float) == False:
            value = np.asscalar(value)
        result_file.write("trait: " + str(key) + ", R square: " + str(value) + "\n")
        total += value
    result_file.write("Average R square: " + str(total / len(results)) + "\n")


def get_best_and_worst(all_outputs, ds, percentile_size=10):
  """Gathers labels/predictions etc for the top and bottom given percentile"""
  traits = all_outputs.keys()
  out = {}
  for trait in traits:
    out[trait] = {}

    l1_errors = np.asarray(all_outputs[trait]['l1_errors'])
    if len(l1_errors) == 0:
      continue

    predictions = np.asarray(all_outputs[trait]['predictions'])
    labels = np.asarray(all_outputs[trait]['labels'])
    additional = all_outputs[trait]['additional']
    # print(additional)

    best_indices = np.where(l1_errors <= np.percentile(l1_errors, percentile_size))[0]
    worst_indices = np.where(l1_errors >= np.percentile(l1_errors, 100-percentile_size))[0]

    out[trait] = {
      'best': {
        'percentile': 100-percentile_size,
        'errors': [l1_errors[i] for i in best_indices],
        'inputs': [ds[i] for i in best_indices],
        'labels': [labels[i] for i in best_indices],
        'predictions': [predictions[i] for i in best_indices],
      },
      'worst': {
        'percentile': percentile_size,
        'errors': [l1_errors[i] for i in worst_indices],
        'inputs': [ds[i] for i in worst_indices],
        'labels': [labels[i] for i in worst_indices],
        'predictions': [predictions[i] for i in worst_indices],
      },
      'all_predictions': predictions,
      'all_targets': labels
    }
    for k, v in additional.items():
      out[trait]['best'][k] = [v[i] for i in best_indices]
      out[trait]['worst'][k] = [v[i] for i in worst_indices]
  return out

if __name__ == '__main__':
    c = config.build_config()
    c.variable_length_params = config.dotdict(config.DEFAULT_VARIABLE_LENGTH_PARAMS)
    print("CONFIG:")
    print(c.json_dumps())

    torch.backends.cudnn.benchmark = True

    result_dir = os.path.join(c.result_dir, c.exp_desc)
    model_path = os.path.join(result_dir, 'model.pth.tar')
    filter_str = ''
    if c.data_filter is not None:
      prefix = '' if c.data_filter[0] == 'col_val' else 'not'
      filter_str = '_' + prefix + c.data_filter[1]['value_match_str']
    results_path = os.path.join(result_dir, f'eval_results_{c.dataset}{filter_str}.bin')
    results_corr_path = os.path.join(result_dir, f'eval_results_corr_{c.dataset}{filter_str}.bin')
    predictions_path = os.path.join(result_dir, f'eval_preds_{c.dataset}{filter_str}.bin')
    best_and_worst_path = os.path.join(result_dir, f'eval_bestworst_{c.dataset}{filter_str}.bin')

    if not os.path.exists(result_dir):
        raise ValueError("Experiment doesn't exist: " + result_dir)
    if not os.path.exists(model_path):
        raise ValueError("No weights found at this path: " + model_path)

    def make_dict():
      return {}

    print("evaluating multi task..", c.traits)
    input_dim = 2001
    input_channels = 2 if c.include_wavelengths else 1
    output_dim = len(c.traits)
    with torch.no_grad():
      model = load_model(c, input_dim, input_channels, output_dim)

    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(c.device)

    # fix seed
    # np.random.seed(1000)

    steps_lower = 7
    steps_upper = 8
    vl_lower_vals = np.linspace(0.4, c.variable_length_params.max_lower, steps_lower)
    vl_upper_vals = np.linspace(c.variable_length_params.min_upper, 2.4, steps_upper)
    results = defaultdict(make_dict)
    results_corr = defaultdict(make_dict)
    time_per_example_ma = 0
    MA = 0.5

    dataset_loader_traits = c.traits[0] if len(c.traits) == 1 else c.traits
    # Evaluate the model while slicing off the ends of the spectra
    for lower in vl_lower_vals:
      for upper in reversed(vl_upper_vals):
        print("Evaluating between: {:.2f} - {:.2f} {}".format(lower, upper, "Filter: " + filter_str if filter_str else ''))
        vl_params = {'max_lower': lower, 'min_upper': upper, 'random': False}
        test_ds, testloader = multitask.load_ds(c, traits=dataset_loader_traits, dataset=c.test_dataset, data_dir=c.data_dir,
                                                batch_size=c.batch_size, include_wavelengths=c.include_wavelengths,
                                                variable_length_params=vl_params,
                                                dataset_version=c.dataset,
                                                shuffle=False)
        epoch_start_time = time()
        examples_per_epoch = len(testloader) * c.batch_size
        _, test_overall_R_squares, _, _, all_outputs = multitask.compute_results(model, c.traits, test_ds, testloader)


        epoch_end_time = time()
        time_per_example = (epoch_end_time - epoch_start_time) / examples_per_epoch if examples_per_epoch > 0 else 0
        time_per_example_ma = MA * time_per_example_ma + (1 - MA) * time_per_example

        mean_r_square = np.mean([v for v in test_overall_R_squares.values() if np.isfinite(v)])
        test_overall_R_squares['mean'] = mean_r_square
        results[lower][upper] = test_overall_R_squares

        corr_coeffs = {}
        for k, v in all_outputs.items():
          corr_coeffs[k] = v['correlation']
        mean_corr_coeff = np.mean([v for v in corr_coeffs.values() if np.isfinite(v)])
        corr_coeffs['mean'] = mean_corr_coeff
        results_corr[lower][upper] = corr_coeffs

        print("Time per example: {:.3f}sec; examples per second: {:d}; R^2: {:.3f}; Corr: {:.3f}\n"
              .format(time_per_example_ma, int(round(1 / (time_per_example_ma + 1e-6))), mean_r_square, mean_corr_coeff))
      #   break
      # break
    with open(results_path, 'wb') as f:
      pickle.dump(results, f)

    with open(results_corr_path, 'wb') as f:
      pickle.dump(results_corr, f)

    # Do one normal pass to save the outputs
    test_ds, testloader = multitask.load_ds(c, traits=dataset_loader_traits, dataset=c.test_dataset, data_dir=c.data_dir,
                                            batch_size=c.batch_size, include_wavelengths=c.include_wavelengths,
                                            dataset_version=c.dataset,
                                            shuffle=False)
    _, _, _, _, all_outputs = multitask.compute_results(model, c.traits, test_ds, testloader)
    # all_outputs['indices'] = test_ds.split_indices.tolist()
    print(all_outputs.keys(), predictions_path)
    with open(predictions_path, 'wb') as f:
      pickle.dump(all_outputs, f)

    # Dump any model additional outputs
    # _, _, _, _, all_outputs = multitask.compute_results(model, c.traits, test_ds, testloader, with_additional=True)
    # best_and_worst = get_best_and_worst(all_outputs, test_ds)
    # with open(best_and_worst_path, 'wb') as f:
    #   pickle.dump(best_and_worst, f)
