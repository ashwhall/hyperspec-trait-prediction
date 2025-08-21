import os
import copy
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
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
        if not isinstance(value, float):
            value = value.item()
        result_file.write(f"trait: {key}, R square: {value}\n")
        total += value
    result_file.write(f"Average R square: {total / len(results)}\n")


def main():
    c = config.build_config()

    result_dir = os.path.join(c.result_dir, c.exp_desc)

    if os.path.exists(result_dir):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_dir += f"_{timestamp}"

    result_file_name = os.path.join(result_dir, "results.txt")
    log_dir = os.path.join(result_dir, "tensorboard")

    os.makedirs(result_dir, exist_ok=True)

#    result_dir = os.path.join(c.result_dir, c.exp_desc)

#    result_file_name = os.path.join(result_dir, "results.txt")
#    log_dir = os.path.join(result_dir, "tensorboard")

#    if not os.path.exists(result_dir):
#        os.makedirs(result_dir)
#    else:
#        raise ValueError(f'Experiment already exists, please give a new description\n({result_dir})')

    writer = SummaryWriter(log_dir=log_dir)

    # Result holders
    test_overall_results = {}
    test_results = {}
    test_bias_results = {}
    test_REP_results = {}
    best_validation_results = {}
    best_validation_bias_results = {}
    best_validation_REP_results = {}
    best_train_results = {}

    # Set loss function
    #criterion_options = {"MSE": nn.MSELoss,
    #                     "L1": nn.L1Loss,
    #                     "Huber": nn.SmoothL1Loss}
    #criterion = criterion_options[c.loss_fn]

    if c.loss_func == "MSE":
        criterion = nn.MSELoss()
    elif c.loss_func == "L1":
        criterion = nn.L1Loss()
    elif c.loss_func == "Huber":
        criterion = nn.SmoothL1Loss()
    else:
        raise ValueError("Wrong loss function specified")

    # Flipped condition to ensure the user wants multi-task
    if not c.multi_task:
        raise ValueError("Must set c.multi_task = True if running train_model_tn.py")

    # CUDA initialisation diagnostics ## was not in the NAbeel's one
    import time
    t0 = time.time()
    print("cuda avail:", torch.cuda.is_available())
    torch.cuda.current_device()         # create context
    torch.empty(1, device='cuda')       # touch device
    torch.cuda.synchronize()
    print("CUDA init took:", round(time.time()-t0, 2), "s")

    # Define predicted traits beforehand to improve logging
    predicted_labels = {}
    for trait in c.traits:
        predicted_labels[trait] = pd.DataFrame(columns=['observed', 'predicted'])
    print("training multi task..", c.traits)

    # Define layer sizes
    input_dim = 2001
    input_channels = 2 if c.include_wavelengths else 1
    output_dim = len(c.traits)
    print(f"working with sizes: in_dim:{input_dim}, in_ch={input_channels}, out_dim={output_dim}")

    # build model
    model = load_model(c, input_dim, input_channels, output_dim)
    print(f"model loaded")

    print([p.requires_grad for n,p in model.named_parameters()])
    print('params checked')

    # apply checkpoint
    if c.load_pretrained_model:
        print(f"Loading pre-trained model weights from: {c.load_pretrained_model}")
        pretrained_dict = torch.load(c.load_pretrained_model)
        model_dict = model.state_dict()

        # Skip incompatible layers like the final fully connected ones
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if "fc_layer.6" not in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("Successfully loaded pre-trained weights for fine-tuning.")

        # Freeze feature extraction layers ONLY
        print("Freezing feature extractor layers. Keeping output layers trainable.")
        for name, param in model.named_parameters():
            if any(x in name.lower() for x in ["cnn", "backbone", "features"]):
                param.requires_grad = False
            else:
                param.requires_grad = True

    # Print which layers are trainable
    print("\nTrainable parameters:")
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad = {param.requires_grad}")

    # move model to gpu
    print(f"moving model to dev {c.device}")
    model.to(c.device)
    print(f"model to dev")

    # dump all the config and model details
    print("CONFIG:")
    print(c.json_dumps())

    print('model summary')
    print(type(model))
    model.summary((input_channels, input_dim))

    # Only optimize trainable parameters
    print('setup optimiser')
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=c.lr,
        weight_decay=c.weight_decay
    )

    # Define traits to use in dataloader
    dataset_loader_traits = c.traits[0] if len(c.traits) == 1 else c.traits

    # === Training ===
    best_overall_model, best_models, train_results, best_val_R_squares, best_val_bias, best_val_rep = multitask.train_multitask_learning_v2(
        c,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        traits=dataset_loader_traits,
        data_dir=c.data_dir,
        batch_size=c.batch_size,
        num_epochs=c.num_epochs,
        include_wavelengths=c.include_wavelengths,
        writer=writer,
        variable_length_params=c.variable_length_params
    )

    # === Evaluation ===
    test_ds, testloader = multitask.load_ds(c, traits=dataset_loader_traits, dataset="test",
                                            data_dir=c.data_dir, batch_size=c.batch_size,
                                            include_wavelengths=c.include_wavelengths)

    _, test_overall_R_squares, test_overall_bias, test_overall_rep, _ = multitask.compute_results(
        best_overall_model, c.traits, test_ds, testloader)

    test_overall_results.update(test_overall_R_squares)
    test_bias_results.update(test_overall_bias)
    test_REP_results.update(test_overall_rep)

    # Trait-specific models
    for trait in c.traits:
        _, test_R_squares = multitask.compute_r_square(best_models[trait], c.traits, test_ds, testloader)
        test_results[trait] = test_R_squares[trait]

    best_train_results.update(train_results)
    best_validation_results.update(best_val_R_squares)
    best_validation_bias_results.update(best_val_bias)
    best_validation_REP_results.update(best_val_rep)

    # Predict & store
    p_labels = multitask.predict(best_overall_model, c.traits, test_ds, testloader)
    for trait in c.traits:
        predicted_labels[trait] = pd.concat([predicted_labels[trait], p_labels[trait]], ignore_index=True)

    # === Results Logging ===
    print("\nResults (Overall model):")
    total = 0.0
    for trait in c.traits:
        val = test_overall_results[trait].item()
        print(f" Trait: {trait:8s} | test R²: {val:.4f}")
        total += val
    print(f"Avg test R² (overall model): {total / len(c.traits):.4f}")

    print("\nResults (Best model per trait):")
    total = 0.0
    for trait in c.traits:
        val = test_results[trait].item() if isinstance(test_results[trait], np.ndarray) else test_results[trait]
        print(f" Trait: {trait:8s} | test R²: {val:.4f}")
        total += val
    print(f"Avg test R² (best per trait): {total / len(c.traits):.4f}")

    # Save model & results
    print("Saving model to:", result_dir)
    torch.save(best_overall_model.state_dict(), os.path.join(result_dir, 'model.pth'))

    with open(result_file_name, "w") as result_file:
        result_file.write("-------------------------------------------------\n")
        result_file.write("Here are the training data set results:\n")
        write_out_results(result_file, best_train_results)
        result_file.write("-------------------------------------------------\n")
        result_file.write("Here are the best validation data set results:\n")
        write_out_results(result_file, best_validation_results)
        result_file.write("-------------------------------------------------\n")
        result_file.write("Here are the test data set results:\n")
        write_out_results(result_file, test_overall_results)

    c.write_to_json_file(os.path.join(result_dir, "commandline_args.txt"))
    writer.close()

import traceback

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print("\n"*2, "Exception occurred during training:", e, '\n'*2)
        print(traceback.format_exc())
