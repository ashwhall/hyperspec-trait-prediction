import argparse
import os
import sys
import yaml
import ast
from copy import deepcopy as deepcopy
import json

import torch
import numpy as np

from dataset.constants import TRAITS


traits_list_str = '[' + ','.join(['"' + t + '"' for t in TRAITS]) + ']'


class dotdict(dict):
  """Adds dot access to dict class: d.abc == d['abc']"""
  __getattr__ = dict.get
  __setattr__ = dict.__setitem__
  __delattr__ = dict.__delitem__
  def __deepcopy__(self, memo=None):
    return dotdict(deepcopy(dict(self), memo=memo))
  def json_dumps(self):
    # Handle non-serialisable items such as the torch.device entry
    def to_str(v): return str(v)
    conf = {k: str(v) for (k, v) in self.items()}
    return json.dumps(conf, indent=2, default=to_str)
  def write_to_json_file(self, file_path):
    with open(file_path, 'w') as f:
      json.dump(self.json_dumps(), f, indent=2)


REQ_ARGS = [
  'exp_desc',
]

ARGS_TO_PARSE = [
  'conv_dilations',
  'augmentation',
  'data_filter',
  'traits'
]

DEFAULT_VARIABLE_LENGTH_PARAMS = {
  'min_width': 0.35, 'max_lower': 0.7, 'min_upper': 1.0,
  'lower_std': 0.1, 'upper_std': 0.5
}
DEFAULT_AUGMENTATION = [
  ['shift_leftright', 0.5, 5],
]

def arg_parse():
    parser = argparse.ArgumentParser(description='Train Model Arg Parser')
    parser.add_argument("--config_file", dest="config_file", help="Config file path", type=str)
    parser.add_argument("--batch_size", dest="batch_size", help="Batch size", default=8, type=int)
    parser.add_argument("--exp_desc", dest="exp_desc", help="experiment description", type=str)
    parser.add_argument("--num_epochs", dest="num_epochs", help="number of epochs", default = 1000, type=int)
    parser.add_argument("--lr", dest="lr", help="learning rate", default=0.0001, type=float)
    parser.add_argument("--model_type", dest="model_type", help="model types: OneDCNN, Linear, LSTM, DeepOneDCNN",type=str)
    parser.add_argument("--validate_every", dest = "validate_every", help = "validate the model every specified epochs", default=10, type=int)
    parser.add_argument("--dropout_percent", dest="dropout_percent", help="percentage of dropout", default=0.2, type=float)
    parser.add_argument("--data_dir", dest="data_dir", help="data directory", default="/data",type=str)
    parser.add_argument("--gpu_device", dest="gpu_device", help="gpu device default cuda:0", default="cuda:0",type=str)
    parser.add_argument("--result_dir", dest="result_dir", help="result directory", default="../results",type=str)
    parser.add_argument("--avg_pool", dest="avg_pool", help="boolean specifying if average pool will be used", action='store_true', default=True)
    parser.add_argument("--stddev", dest="stddev", help="gaussian stddev", default=0.0, type=float)
    parser.add_argument("--weight_decay", dest="weight_decay", help="weight_decay", default=0.0001, type=float)
    parser.add_argument("--LSTM_avg_pool_width", dest="LSTM_avg_pool_width", help = "the average pool width for LSTM", default=10, type=int)
    parser.add_argument("--CNN_avg_pool_width", dest="CNN_avg_pool_width", help = "the average pool width for CNN", default=10, type=int)
    parser.add_argument("--CNN_kernel_size", dest="CNN_kernel_size", help="CNN kernel size", default=5, type=int)
    parser.add_argument("--CNN_dilation", dest="CNN_dilation", help="CNN dilation factor", default=2, type=int)
    parser.add_argument("--CNN_num_channels", dest="CNN_num_channels", help="CNN number of output channels", default=50, type=int)
    parser.add_argument("--LSTM_hidden_dim", dest="LSTM_hidden_dim", help="LSTM hidden units dimension", default=50, type=int)
    parser.add_argument("--LSTM_number_layers", dest="LSTM_number_layers", help="LSTM number layers", default=2, type=int)
    parser.add_argument("--multi_task", dest="multi_task", help="Boolean specifying multi-task learning", action='store_true', default=False)
    parser.add_argument("--LSTM_bidirectional", dest="LSTM_bidirectional", help ="LSTM boolean specifying if bidirectional will be used", action='store_true', default=True)
    parser.add_argument("--pre_train_num_epochs", dest="pre_train_num_epochs", help="number of epochs for pre-training", default=500, type=int)
    parser.add_argument("--loss_func", dest="loss_func", help="the loss function: MSE, Huber, L1", default="MSE", type=str)
    parser.add_argument("--norm", dest="norm", help="the normalization method: batch_norm, layer_norm", default = "batch_norm", type=str)
    parser.add_argument("--data_aug_type", dest="data_aug_type", help="data_augmentation type", default="None", type=str)
    parser.add_argument("--data_aug_percent", dest="data_aug_percent", help="data augmentation percent", default=0.0, type=float)
    parser.add_argument("--data_aug_rate", dest="data_aug_rate", help="data augmentation rate", default=0.0, type=float)
    parser.add_argument("--dataset", dest="dataset", help="dataset version", default="v1", type=str)
    parser.add_argument("--include_wavelengths", dest="include_wavelengths", default=False, help="Stack the wavelength with frequency response for inputs", action='store_true')
    parser.add_argument("--variable_length", dest="variable_length", default=False, help="If True, the length of the spectra varies per sample", action="store_true")
    parser.add_argument("--conv_dilations", dest="conv_dilations", default=None, help="List of dilation rates for conv layers - as a string", type=str)
    parser.add_argument("--augmentation", dest="augmentation", default=None, help="List of augmentation parameters - as a string", type=str)
    parser.add_argument("--CNN_interm_channels", dest="CNN_interm_channels", help="Number of intermediate channels for CNN", default=32, type=int)
    parser.add_argument("--CNN_attention", dest="CNN_attention", default=False, help="Whether to use attention mechanism in CNN", action='store_true')
    parser.add_argument("--squeeze_ratio", dest="squeeze_ratio", help="Squeeze ratio for Squeeze-Excitation model", default=8, type=int)
    parser.add_argument("--no_rng_seed", dest="no_rng_seed", default=False, help="Whether to use seed the RNGs", action='store_true')
    parser.add_argument("--max_iter", dest="max_iter", help="Maximum number of iterations for PLS algorithm", default=1000, type=int)
    parser.add_argument("--data_filter", dest="data_filter", default=None, help="The parameters of any data filtering", type=str)
    parser.add_argument("--traits", dest="traits", default=traits_list_str, help="The traits used in the experiment - as a string", type=str)
    parser.add_argument("--test_dataset", dest="test_dataset", default="test", help="Which dataset partition to test models on - should probably be either 'test' or 'all'", type=str)
    parser.add_argument("--model_summary", dest="model_summary", help="Prints model summary when used with config.py.__main__", default=False, action='store_true')
    parser.add_argument("--rng_seed", dest="rng_seed", help="Set a specific seed value", default=1000, type=int)

    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    args.device = torch.device(args.gpu_device if use_cuda else "cpu")
    return args


def deep_merge(source, destination):
  """Deep copy the values from source to destination dicts"""
  for key, value in source.items():
    if isinstance(value, dict):
      node = destination.setdefault(key, {})
      deep_merge(value, node)
    else:
      destination[key] = value

  return destination


def load_config_file(path):
  """Reads a config fule and returns the dict"""
  if os.path.exists(path):
    with open(path) as file:
      loaded = dotdict(yaml.load(file, Loader=yaml.FullLoader))
  else:
    raise FileNotFoundError("Config file not found: {}".format(path))

  # Inject the config parameters into argv then parse again
  # This is a hacky way to override the config file with args
  for k, v in loaded.items():
    if not isinstance(v, bool):
      sys.argv.insert(1, str(v))
    # If it is a bool only insert if it's true
    if v:
      sys.argv.insert(1, f'--{k}')
  args = arg_parse()
  return dotdict(deep_merge(args.__dict__, loaded.__dict__))


def check_required_args(c):
  """Throws exception if a required arg is missing from the digen dict"""
  missing_args = []
  for req_arg in REQ_ARGS:
    if not c.get(req_arg):
      missing_args.append(req_arg)
  if missing_args:
    raise ValueError("Missing arguments: {}".format(missing_args))


def delete_nones(d):
  """Deeply deletes all Nones from a given dict"""
  changed = True
  while changed:
    changed = False
    for key, value in d.items():
      if value is None or (isinstance(value, str) and len(value) == 0):
        del d[key]
        changed = True
        break
      elif isinstance(value, dict):
        delete_nones(value)


def parse_args_strings(d):
  """Any args that we expect to be python literal are parsed to their values"""
  for k in ARGS_TO_PARSE:
    v = d.get(k, False)
    if v and not isinstance(v, list):
      d[k] = ast.literal_eval(v)

def seed_rngs(seed):
  """Seed random number generators"""
  torch.random.manual_seed(seed)
  np.random.seed(seed)


def finalise_config(c):
  delete_nones(c)
  parse_args_strings(c)
  # check_required_args(c)

  c.variable_length_params = dotdict(DEFAULT_VARIABLE_LENGTH_PARAMS) if c.variable_length else {}
  return c


def build_config():
  """
  Build the config file using a combination of command-line args and the config file.
  The command-line args take precedence over the config file.
  """
  args = arg_parse()
  c = dotdict(args.__dict__)

  if c.get('config_file'):
    c = load_config_file(c.config_file)

  c = finalise_config(c)

  if not c.no_rng_seed:
    seed_rngs(c.rng_seed)
  # torch.backends.cudnn.benchmark = True
  return c


if __name__ == '__main__':
  print("Config contents:")
  import pprint
  c = build_config()
  pprint.pprint(c)

  if c.model_summary:
      from models.utils import load_model
      input_dim = 2001
      input_channels = 2 if c.include_wavelengths else 1
      output_dim = len(c.traits)
      model = load_model(c, input_dim, input_channels, output_dim)
      model.to(c.device)

      model.summary((input_channels, input_dim))
