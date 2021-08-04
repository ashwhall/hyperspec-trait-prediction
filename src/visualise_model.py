import os
import pickle
import torch
import numpy as np

import config
import multitask_learning as multitask
from models.utils import load_model


if __name__ == '__main__':
    c = config.build_config()
    print("CONFIG:")
    print(c.json_dumps())

    torch.backends.cudnn.benchmark = True

    result_dir = os.path.join(c.result_dir, c.exp_desc)
    model_path = os.path.join(result_dir, 'model.pth.tar')
    print("Loading model from", result_dir)
    if not os.path.exists(result_dir):
        raise ValueError("Experiment doesn't exist!")
    if not os.path.exists(model_path):
        raise ValueError("No weights found at this path")

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



    dataset_loader_traits = c.traits[0] if len(c.traits) == 1 else c.traits
    # Evaluate the model while slicing off the ends of the spectra
    test_ds, testloader = multitask.load_ds(c, traits=dataset_loader_traits, dataset="test", data_dir=c.data_dir,
                                            batch_size=c.batch_size, include_wavelengths=c.include_wavelengths,
                                            dataset_version=c.dataset)
                                            # shuffle=False)

    sequential = list(model.children())[1]
    block = list(sequential.children())[0]
    conv = list(list(block.children())[0])[0]


    activations = []
    def print_tensor_props(self, input, output):
      mean_activations = torch.tanh(torch.mean(output, 0)).cpu().detach().numpy()
      activations.append(mean_activations)

    conv.register_forward_hook(print_tensor_props)
    with torch.no_grad():
      for i, data in enumerate(testloader, 0):
          values = data["value"].to(c.device)


          traits_outputs = model(values)
    activations = np.array(activations)
    activations = np.mean(activations, 0)
    print(activations.shape)

    dir_path = os.path.join('bin', 'mean_activations')
    if not os.path.exists(dir_path):
      os.makedirs(dir_path)
    name = os.path.join(dir_path, f'{"+".join(c.traits)}_{c.dataset}.bin')
    print("saving:", name)
    if activations.ndim > 0:
      with open(name, 'wb') as f:
        pickle.dump(activations, f)
