import os
import copy
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tensorboardX import SummaryWriter

import config
from models.utils import load_model
import multitask_learning as multitask


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


def main():
    c = config.build_config()

    result_dir = os.path.join(c.result_dir, c.exp_desc)

    result_file_name = result_dir +"/results.txt"
    log_dir = result_dir + "/tensorboard"

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    else:
        raise ValueError('Experiment already exists, please give new description \n(' + result_dir + ')')

    writer = SummaryWriter(log_dir=log_dir)

    test_overall_results = {} # for best overall model
    test_results = {} # for best model for each trait
    test_bias_results = {}
    test_REP_results = {}

    best_validation_results = {}
    best_validation_bias_results = {}
    best_validation_REP_results = {}

    best_train_results = {}

    if c.loss_func == "MSE":
        criterion = nn.MSELoss()
    elif c.loss_func == "L1":
        criterion = nn.L1Loss()
    elif c.loss_func == "Huber":
        criterion = nn.SmoothL1Loss()
    else:
        raise ValueError('Wrong loss function specification')

    if c.multi_task:
        predicted_labels = {}
        for trait in c.traits:
            predicted_labels[trait] = pd.DataFrame(columns=['observed', 'predicted'])

        print("training multi task..", c.traits)

        input_dim = 2001
        input_channels = 2 if c.include_wavelengths else 1
        output_dim = len(c.traits)
        model = load_model(c, input_dim, input_channels, output_dim)
        model.to(c.device)

        print("CONFIG:")
        print(c.json_dumps())

        model.summary((input_channels, input_dim))

        optimizer = optim.Adam(model.parameters(), lr=c.lr, weight_decay=c.weight_decay)
        # fix seed
        # np.random.seed(1000)

        # Handle single trait training
        dataset_loader_traits = c.traits[0] if len(c.traits) == 1 else c.traits
        best_overall_model, best_models, train_results, best_val_R_squares, best_val_bias, best_val_rep = multitask.train_multitask_learning_v2(c,
                                                                                                                    model=model,
                                                                                                                    criterion=criterion,
                                                                                                                    optimizer=optimizer,
                                                                                                                    traits=dataset_loader_traits,
                                                                                                                    data_dir=c.data_dir,
                                                                                                                    batch_size=c.batch_size,
                                                                                                                    num_epochs=c.num_epochs,
                                                                                                                    include_wavelengths=c.include_wavelengths,
                                                                                                                    writer=writer,
                                                                                                                    variable_length_params=c.variable_length_params)
        test_ds, testloader = multitask.load_ds(c, traits=dataset_loader_traits, dataset="test", data_dir=c.data_dir,
                                                batch_size=c.batch_size, include_wavelengths=c.include_wavelengths)

        _, test_overall_R_squares, test_overall_bias, test_overall_rep, _ = multitask.compute_results(best_overall_model, c.traits, test_ds, testloader)
        test_overall_results.update(test_overall_R_squares)
        test_bias_results.update(test_overall_bias)
        test_REP_results.update(test_overall_rep)

        # use best model for each trait
        for trait in c.traits:
            _, test_R_squares = multitask.compute_r_square(best_models[trait], c.traits, test_ds,
                                                           testloader)
            test_results[trait] = test_R_squares[trait]

        best_train_results.update(train_results)
        best_validation_results.update(best_val_R_squares)
        best_validation_bias_results.update(best_val_bias)
        best_validation_REP_results.update(best_val_rep)
        p_labels = multitask.predict(best_overall_model, c.traits, test_ds, testloader)
        for trait in c.traits:
            predicted_labels[trait] = predicted_labels[trait].append(p_labels[trait])

        total = 0.0
        print("Overall model")
        for trait in c.traits:
            print(" Trait: {:s} test R square: {:.4f} ".format(trait, np.asscalar(test_overall_results[trait])))
            total += np.asscalar(test_overall_results[trait])
        print("Avg test R square: {:.4f} ".format(total / len(test_overall_results)))

        total = 0.0
        print("best model for each trait")
        for trait in c.traits:
          if np.isfinite(test_results[trait]):
            value = test_results[trait].item() if isinstance(test_results[trait], np.ndarray) else test_results[trait]
            print(" Trait: {:s} test R square: {:.4f} ".format(trait, value))
            total += value
        print("Avg test R square: {:.4f} ".format(total / len(test_results)))

        print("Saving to", result_dir)
        torch.save(best_overall_model.state_dict(), os.path.join(result_dir, 'model.pth.tar'))
    else:
        raise ValueError("Must set c.multi_task = True if running train_model_tn.py")

    writer.close()

    result_file = open(result_file_name,"w")
    result_file.write("-------------------------------------------------\n")
    result_file.write("Here are the training data set results:\n")
    write_out_results(result_file, best_train_results)
    result_file.write("-------------------------------------------------\n")
    result_file.write("Here are the best validation data set results:\n")
    write_out_results(result_file, best_validation_results)
    result_file.write("-------------------------------------------------\n")
    result_file.write("Here are the test data set results:\n")
    write_out_results(result_file, test_overall_results)
    result_file.close()

    c.write_to_json_file(result_dir + "/commandline_args.txt")

if __name__ == '__main__':
  try:
    main()
  except Exception as e:
    print(e)
    pass
