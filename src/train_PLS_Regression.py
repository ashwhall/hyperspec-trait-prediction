import config
import dataset.utils as utils
import dataset.HyperSpectrumDataSet as hsd
import os
import pickle
import copy

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.cross_decomposition import PLSRegression



DATASETS = list(utils.FILE_PATHS.keys())

# Set to true if we want different config for test
SPECIAL_TEST = False

def convert_df(data_dict, column_name):
    df = pd.DataFrame.from_dict(data_dict, orient='index')
    df = df.rename(index=str, columns={0: column_name})
    return df

def write_out_results(result_file, results, metric_name='R square'):
    total = 0.0
    for key, value in results.items():
        value = float(value)
        result_file.write("trait: " + str(key) + f", {metric_name}: " + str(value) + "\n")
        total += value
    result_file.write(f"Average {metric_name}: " + str(total / len(results)) + "\n")

def compute_bias(observed, predicted):
    mean_observed = np.mean(observed)
    mean_predicted = np.mean(predicted)

    bias = 100.0 * (mean_predicted - mean_observed) / mean_observed

    return bias

def compute_rep(observed, predicted):
    mean_observed = np.mean(observed)
    num = len(observed)
    minus = observed - predicted
    square = np.square(minus)
    sum = np.sum(square)
    devideLen = sum / num
    sqrt = np.sqrt(devideLen)
    rep = 100.0 * sqrt / mean_observed
    return rep

def test_on_dataset(models, c, dataset, traits, result_dir, split='test'):
  print("==================== Full test on", dataset, "====================")
  result_file_name = result_dir + f"/test_results_{dataset}.txt"
  corr_result_file_name = result_dir + f"/test_corr_results_{dataset}.txt"
  all_outputs_file_name = result_dir + f"/all_outputs_{dataset}.bin"

  c = copy.deepcopy(c)
  c.data_filter = None
  test_results = {}
  corr_results = {}
  all_outputs = {}
  for trait in traits:
    test_ds = hsd.HyperSpectrumDataSet(c, trait, split, c.data_dir, dataset=dataset)
    if len(test_ds) == 0 or trait not in models:
      print("SKIPPING TRAIT:", trait)
      all_outputs[trait] = {
        'targets'    : np.array([]),
        'predictions': np.array([]),
        'indices'    : np.array([])
      }
      test_results[trait] = np.nan
      corr_results[trait] = np.nan
      continue

    X_test, y_test = test_ds.data_values, test_ds.data_labels
    y_test = y_test.values.ravel()

    test_y_pred = models[trait].predict(X_test)
    all_outputs[trait] = {
      'targets': test_ds.denormalize(y_test),
      'predictions': test_ds.denormalize(test_y_pred),
      'indices': test_ds.split_indices
    }
    test_y_pred = test_y_pred.ravel()
    test_rsquare = r2_score(y_test, test_y_pred)
    test_results[trait] = test_rsquare
    corr_results[trait] = np.corrcoef(y_test, test_y_pred)[0, 1]

  with open(result_file_name, "w") as result_file:
    result_file.write("-------------------------------------------------\n")
    result_file.write("n_components: " + str(n_components) + "\n")
    result_file.write("Test data set results:\n")
    write_out_results(result_file, test_results)
  with open(corr_result_file_name, "w") as result_file:
    result_file.write("-------------------------------------------------\n")
    result_file.write("n_components: " + str(n_components) + "\n")
    result_file.write("Test data set results:\n")
    write_out_results(result_file, corr_results, 'correlation')
  with open(all_outputs_file_name, 'wb') as f:
    pickle.dump(all_outputs, f)

def test_on_aus_mex(models, c, traits, result_dir):
  dataset = 'v1'

  for filter_val in ['Aus', 'Mex']:
    c = copy.deepcopy(c)
    c.data_filter = ('col_val', {'column_name': 'Exp', 'value_match_str': '' + filter_val + ''})
    print("==================== Full test on " + filter_val + " split ====================")
    result_file_name = result_dir + f"/test_results_{dataset}_{filter_val}.txt"
    corr_result_file_name = result_dir + f"/test_corr_results_{dataset}_{filter_val}.txt"

    test_results = {}
    corr_results = {}
    for trait in traits:
      test_ds = hsd.HyperSpectrumDataSet(c, trait, c.test_dataset, c.data_dir, dataset=dataset)
      if len(test_ds) == 0 or trait not in models:
        print("SKIPPING TRAIT:", trait)
        test_results[trait] = np.nan
        continue

      X_test, y_test = test_ds.data_values, test_ds.data_labels
      y_test = y_test.values.ravel()

      test_y_pred = models[trait].predict(X_test)
      test_y_pred = test_y_pred.ravel()
      test_rsquare = r2_score(y_test, test_y_pred)
      test_results[trait] = test_rsquare
      corr_results[trait] = np.corrcoef(y_test, test_y_pred)[0, 1]

    with open(result_file_name, "w") as result_file:
      result_file.write("-------------------------------------------------\n")
      result_file.write("n_components: " + str(n_components) + "\n")
      result_file.write("Test data set results:\n")
      write_out_results(result_file, test_results)
    with open(corr_result_file_name, "w") as result_file:
      result_file.write("-------------------------------------------------\n")
      result_file.write("n_components: " + str(n_components) + "\n")
      result_file.write("Test data set results:\n")
      write_out_results(result_file, corr_results, 'correlation')


if __name__ ==  '__main__':
    c = config.build_config()

    train_results = {}
    val_results = {}
    test_results = {}
    val_bias_results = {}
    val_rep_results = {}
    test_bias_results = {}
    test_rep_results = {}
    predicted_labels = {}
    train_sum = 0
    test_sum = 0

    result_dir = os.path.join(c.result_dir, c.exp_desc)

    result_file_name = result_dir + "/results.txt"

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    else:
        raise ValueError('Experiment already exists please give new description')

    print("Training " + c.exp_desc + "..")
    models = {}
    for target_trait in c.traits:
        print("\nTrait:", target_trait)

        train_ds = hsd.HyperSpectrumDataSet(c, target_trait, "train", c.data_dir, dataset=c.dataset)
        if len(train_ds) == 0:
          print("SKIPPING", target_trait, "due to no training examples")
          continue

        val_ds = hsd.HyperSpectrumDataSet(c, target_trait, "val", c.data_dir, dataset=c.dataset)
        test_dict = c
        if SPECIAL_TEST:
          import copy
          test_dict = copy.deepcopy(c)
          test_dict['data_filter'][1]['value_match_str'] = 'Mex'
        test_ds = hsd.HyperSpectrumDataSet(test_dict, target_trait, "test", c.data_dir, dataset=c.dataset)
        X_train, y_train = train_ds.data_values, train_ds.data_labels
        X_test, y_test = test_ds.data_values, test_ds.data_labels
        X_val, y_val = val_ds.data_values, val_ds.data_labels

        y_train = y_train.values.ravel()
        y_test = y_test.values.ravel()
        y_val = y_val.values.ravel()

        best_rsquare = -999999.99
        val_bias = -999999.99
        val_rep = -999999.99
        best_pc = -1
        for i in np.arange(1, min(X_train.shape[1], 30)):
            pls = PLSRegression(n_components=i, max_iter=c.max_iter)
            pls.fit(X_train, y_train)

            # Prediction
            val_y_pred = pls.predict(X_val)
            # Calculate scores
            val_rsquare = r2_score(y_val, val_y_pred)

            if best_rsquare < val_rsquare:
                best_rsquare = val_rsquare
                best_pc = i
        n_components = best_pc
        print("n_components found:", n_components)
        pls = PLSRegression(n_components=n_components, max_iter=c.max_iter)
        pls.fit(X_train, y_train)

        # Prediction
        train_y_pred = pls.predict(X_train)
        val_y_pred = pls.predict(X_val)
        test_y_pred = pls.predict(X_test)

        models[target_trait] = pls

        train_y_pred = train_y_pred.ravel()
        val_y_pred = val_y_pred.ravel()
        test_y_pred = test_y_pred.ravel()

        # Calculate scores
        train_rsquare = r2_score(y_train, train_y_pred)
        val_rsquare = r2_score(y_val, val_y_pred)
        test_rsquare = r2_score(y_test, test_y_pred)

        # denormalize
        y_val = train_ds.denormalize(y_val)
        val_y_pred = train_ds.denormalize(val_y_pred)
        y_test = train_ds.denormalize(y_test)
        test_y_pred = train_ds.denormalize(test_y_pred)

        val_bias = compute_bias(y_val, val_y_pred)
        val_rep = compute_rep(y_val, val_y_pred)
        test_bias = compute_bias(y_test, test_y_pred)
        test_rep = compute_rep(y_test, test_y_pred)
        print(target_trait, val_rep, test_rep)
        train_results[target_trait] = train_rsquare
        val_results[target_trait] = val_rsquare
        test_results[target_trait] = test_rsquare
        val_bias_results[target_trait] = val_bias
        val_rep_results[target_trait] = val_rep
        test_bias_results[target_trait] = test_bias
        test_rep_results[target_trait] = test_rep

        print(target_trait, train_rsquare, val_rsquare, test_rsquare)
        y_test_df = pd.DataFrame(y_test)
        y_test_df.rename(columns={0: 'test_y'}, inplace=True)
        test_y_pred_df = pd.DataFrame(test_y_pred)
        test_y_pred_df.rename(columns={0: 'test_y_pred'}, inplace=True)

        predicted_labels[target_trait] = pd.concat([y_test_df, test_y_pred_df], axis=1)

    for dataset in ['v1']:
      # Use the test split for any datasets in our training set, else use all
      test_on_dataset(models, c, dataset, c.traits, result_dir, split='test' if c.dataset in dataset else c.test_dataset)

    # print report to file
    result_file = open(result_file_name, "w")
    result_file.write("-------------------------------------------------\n")
    result_file.write("n_components: " + str(n_components) +"\n")
    result_file.write("Here are the training data set results:\n")
    write_out_results(result_file, train_results)
    result_file.write("-------------------------------------------------\n")
    result_file.write("Here are the best validation data set results:\n")
    write_out_results(result_file, val_results)
    result_file.write("-------------------------------------------------\n")
    result_file.write("Here are the test data set results:\n")
    write_out_results(result_file, test_results)
    result_file.close()

    train_df = convert_df(train_results, "train_rsquare")
    best_val_df = convert_df(val_results, "best_val_rsquare")
    test_df = convert_df(test_results, "test_rsquare")
    val_bias_df = convert_df(val_bias_results, "val_bias")
    val_rep_df = convert_df(val_rep_results, "val_rep")
    test_bias_df = convert_df(test_bias_results, "test_bias")
    test_rep_df = convert_df(test_rep_results, "test_rep")

    merged_df = pd.concat([train_df, best_val_df, test_df, val_bias_df, val_rep_df, test_bias_df, test_rep_df], axis=1)
    merged_df = merged_df.rename_axis("trait")

    results_csv = result_dir + "/results.csv"
    merged_df.to_csv(results_csv)

    for trait in c.traits:
        results_csv = result_dir + "/predict_labels_"+trait+".csv"
        if trait in predicted_labels:
          predicted_labels[trait].to_csv(results_csv, index=False)

    with open(os.path.join(result_dir, 'models.bin'), 'wb') as f:
      pickle.dump(models, f)
