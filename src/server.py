import os
import sys
import json
import pickle
from abc import ABC, abstractmethod

from flask import Flask, request, Response, send_from_directory
import numpy as np
import torch
import pandas as pd

import dataset.constants as constants
import models.utils as model_utils
import config


DEV_MODE = os.environ.get('MODE') == 'dev'
DEVICE = torch.device(os.environ.get('DEVICE', 'cpu') if torch.cuda.is_available() else 'cpu')

BASE_PATH = os.getcwd()
if DEV_MODE:
  BASE_PATH = os.path.join(BASE_PATH, 'src')

NUM_TRAITS = len(constants.SERVER_PREDICT_TRAITS)
STAT_LOAD_PATH = os.path.join(constants.DATA_DIR, 'statistics.json')

MAX_BATCH_SIZE = 32
MAX_NUM_SAMPLES = 100

CNN_WL_LOWER = 400
CNN_WL_UPPER = 2400

def _load_statistics(path):
  """Load means and std-devs from the json file and return in a dict"""
  with open(path, 'r') as f:
    stats = json.load(f)
    input_statistics = stats['values']
    trait_statistics = stats['traits']

  return {
    'inputs': {
      'mean': np.array(input_statistics['mean']),
      'std': np.array(input_statistics['std'])
    },
    'traits': {
      'mean': np.array([trait_statistics['mean'][t] for t in constants.SERVER_PREDICT_TRAITS]),
      'std': np.array([trait_statistics['std'][t] for t in constants.SERVER_PREDICT_TRAITS])
    }
  }


STATS = _load_statistics(STAT_LOAD_PATH)

MODELS = {
  'cnn_multi': {
    'config': os.path.join(BASE_PATH, 'config', 'ThiMultiTask.yml'),
    'path': 'MultiCNN.pth.tar',
    'input_dim': 2001,
    'input_channels': 1,
  },
  'cnn_single': {
    'config': os.path.join(BASE_PATH, 'config', 'ThiSingleTask.yml'),
    'path': 'SingleCNN.bin',
    'input_dim': 2001,
    'input_channels': 1,
  }
}
PLS_RANGES = [(400, 900), (400, 1000), (400, 1700), (400, 2400)]
for low, high in PLS_RANGES:
  name = f'pls_{low}-{high}'
  MODELS[name] = {
    'bin': f'{name}.bin',
    'wl_lower': low,
    'wl_upper': high,
  }


def _nearest_pls_range(req_low, req_high):
  """
  Find the best PLS range for which we have a model. 'Best' is determined as one where the
  bounds of the model are within the requests low/high range, with the smallest gap between
  the two ranges.
  E.g. If the request has data in [350, 1200] and we have to choose between models with ranges
    [400, 1000] - distance = |350-400| + |1200-1000| = 250
    [350, 1100] - distance = |350-350| + |1200-1100| = 100
    [300, 1100] - unusable as model range (lower-bound of 300) outside request range (350)
    [400, 1200] - distance = |350-400| + |1200-1200| = 50 <- best range
  """
  best_dist, best_low, best_high = None, None, None
  for low, high in PLS_RANGES:
    if req_low <= low and req_high >= high:
      dist = abs(req_low - low) + abs(req_high - high)
      if best_dist is None or dist < best_dist:
        best_dist, best_low, best_high = dist, low, high

  return best_low, best_high


class HTTPError(Exception):
  def __init__(self, msg, code):
    self.msg, self.code = msg, code
class RequestError(HTTPError):
  """The HTTP request itself was problematic"""
  def __init__(self, msg):
    super().__init__(msg, 400)
class FileFormatError(HTTPError):
  """The CSV file can't be parsed"""
  def __init__(self, msg):
    super().__init__(msg, 422)
class ServerError(HTTPError):
  """Miscellaneous server error"""
  def __init__(self, msg):
    super().__init__(msg, 500)


class Torchify:
  """Class to wrap torch models which handles converting to and from torch.tensor"""
  def __init__(self, model, wl_lower, wl_upper):
    self._model = model
    self._pad_left = CNN_WL_LOWER - wl_lower
    self._pad_right = wl_upper - CNN_WL_UPPER
  def pad_input(self, x):
    if self._pad_left > 0:
      x = x[:, self._pad_left:]
    elif self._pad_left < 0:
      x = np.pad(x, ((0, 0), (-self._pad_left, 0)))

    if self._pad_right > 0:
      x = x[:, :x.shape[-1] - self._pad_right]
    elif self._pad_right < 0:
      x = np.pad(x, ((0, 0), (0, -self._pad_right)))
    return x
  def drop_wl(self, x):
    return x[:, 0]
  def forward(self, x):
    x = self.drop_wl(x)
    x = self.pad_input(x)
    with torch.no_grad():
      x = torch.tensor(x, device=DEVICE).float()
      preds = self._model.forward(x)
      return preds.cpu().detach().numpy()


class SingleTraitWrapper(ABC):
  """
  Abstract wrapper class around a dictionary of models, keyed by traits.
  Implementations need to implement _make_prediction, which describes how a single model makes a prediction,
  as this varies between model types. _process_inputs and _process_predictions can optionally be implemented
  to perform any processing before and after predicting, respectively.
  """
  def __init__(self, models):
    self._models = models
  def _process_inputs(self, x):
    return x
  @abstractmethod
  def _make_prediction(self, model, x):
    pass
  def _process_predictions(self, predictions):
    return predictions
  def forward(self, x):
    predictions = []
    x = self._process_inputs(x)

    for trait in constants.SERVER_PREDICT_TRAITS:
      trait_preds = self._make_prediction(self._models[trait], x)
      predictions.append(trait_preds)
    return self._process_predictions(predictions)


class PLSModels(SingleTraitWrapper):
  def __init__(self, models, req_lower, req_upper, model_lower, model_upper):
    super().__init__(models)
    self._slice_left = model_lower - req_lower
    self._slice_right = (model_upper - req_upper) or None
  def _process_inputs(self, x):
    # Slice the ends off to match the chosen model range and drop the wavelengths
    return x[:, 0, self._slice_left:self._slice_right]
  def _make_prediction(self, model, x):
    return model.predict(x)
  def _process_predictions(self, predictions):
    return np.asarray(predictions)[..., 0].T


class TorchModels(SingleTraitWrapper):
  def _make_prediction(self, model, x):
    return model.forward(x)
  def _process_predictions(self, predictions):
    return torch.cat(predictions, -1)


class Ensemble:
  def __init__(self, models):
    self._models = models
  def forward(self, x):
    predictions = []
    for model in self._models:
      preds = model.forward(x)
      predictions.append(preds)
    predictions = np.mean(predictions, 0)
    return predictions



def reset_argv(): sys.argv = ['/app/src/server.py']
def load_cnn_multi():
  model_info = MODELS['cnn_multi']

  reset_argv()
  c = config.load_config_file(model_info['config'])
  c = config.finalise_config(c)

  model = model_utils.load_model(c, model_info['input_dim'], model_info['input_channels'], NUM_TRAITS)
  model.load_state_dict(torch.load(os.path.join(constants.WEIGHTS_DIR, model_info['path']),
                                   map_location=DEVICE))
  model.eval()
  model.to(DEVICE)
  return model


def load_cnn_single():
  model_info = MODELS['cnn_single']

  reset_argv()
  c = config.load_config_file(model_info['config'])
  c = config.finalise_config(c)

  with open(os.path.join(constants.WEIGHTS_DIR, model_info['path']), 'rb') as f:
    weights = pickle.load(f)

  models = {}
  for trait in constants.SERVER_PREDICT_TRAITS:
    model = model_utils.load_model(c, model_info['input_dim'], model_info['input_channels'], 1)
    model.load_state_dict(weights[trait])
    model.eval()
    model.to(DEVICE)
    models[trait] = model

  return TorchModels(models)


def load_pls(req_lower, req_upper):
  model_lower, model_upper = _nearest_pls_range(req_lower, req_upper)
  model_str = f'pls_{model_lower}-{model_upper}'

  model_info = MODELS.get(model_str, None)
  if not model_info:
    raise FileFormatError(f'The spectral domain [{req_lower}, {req_upper}] is not supported by PLS models')

  with open(os.path.join(constants.WEIGHTS_DIR, model_info['bin']), 'rb') as f:
    models = pickle.load(f)
  models = dict({k: v for k, v in models.items() if k in constants.SERVER_PREDICT_TRAITS})

  return PLSModels(models, req_lower, req_upper, model_lower, model_upper)


def load_ensemble(wl_lower, wl_upper):
  return Ensemble([load_pls(wl_lower, wl_upper),
                   Torchify(load_cnn_multi(), wl_lower, wl_upper),
                   Torchify(load_cnn_single(), wl_lower, wl_upper)])

def _build_model(name, wl_lower, wl_upper):
  if name == 'cnn_multi':
    return Torchify(load_cnn_multi(), wl_lower, wl_upper)
  elif name == 'cnn_single':
    return Torchify(load_cnn_single(), wl_lower, wl_upper)
  elif name == 'pls':
    return load_pls(wl_lower, wl_upper)
  elif name == 'ensemble':
    return load_ensemble(wl_lower, wl_upper)
  raise RequestError('The provided model name is not supported')


def _nice_trait_name(trait):
  if trait.lower().endswith('_o'):
    return trait[:trait.lower().rindex('_o')]
  return trait


def _format_response(raw_preds, names):
  # Bundle the results and return
  out = []
  for trait_values, row_name in zip(raw_preds, names):
    row = {
      'name': row_name
    }
    for t_name, t_val in zip(constants.SERVER_PREDICT_TRAITS, trait_values):
      row[_nice_trait_name(t_name)] = t_val
    out.append(row)
  return out


def jump_correction(stacked, jumps):
  if jumps:
    for data_index in range(0, len(stacked), 1):
      value = stacked[data_index][0]
      wave = stacked[data_index][1]
      for jump in jumps:
        j = jump/1000
        j_1 = (jump-1)/1000
        j1 = (jump+1)/1000
        j2 = (jump+2)/1000
        o_index = np.where(wave == j_1)[0][0]
        o_val = value[o_index]
        o = [j_1, o_val]
        p_index = np.where(wave == j)[0][0]
        p_val = value[p_index]
        p = [j, p_val]
        q_index = np.where(wave == j1)[0][0]
        q_val = value[q_index]
        q = [j1, q_val]
        r_index = np.where(wave == j2)[0][0]
        r_val = value[r_index]
        r = [j2, r_val]
        t1 = (p[1] - o[1])/(p[0] - o[0])
        t2 = (r[1] - q[1])/(r[0] - q[0])
        qqy = ((t1 + t2) * (q[0] - p[0]))/2 + p[1]
        correction = qqy/q[1]
        for wv_sq in range(p_index + 1, len(wave), 1):
          value[wv_sq] = value[wv_sq] * correction
      stacked[data_index][0] = value
  return stacked


def _parse_reflectance_data(csv_file, jumps=None):
  truncated = False
  df = pd.read_csv(csv_file)
  names = list(df.columns)[1:]

  #check whether the file contains all 0 column
  try:
    list_tmp_remove = []
    for n in names:
      if (df[n] == 0).all():
        list_tmp_remove.append(n)
    if len(list_tmp_remove) != 0:
      for col_name in list_tmp_remove:
        del df[col_name]
    names = list(df.columns)[1:]
  except Exception as e:
    print(e)
    # Catch and re-throw all others
    raise FileFormatError('An error occurred while parsing the CSV file - ensure the format matches the example')
  try:
    # This works for 2019-10-22_Population_3_OASS_compiled.csv (file provided by Gonzalo to specify CSV format)
    wavelengths = np.array(df[df.columns[0]])

    if np.any(wavelengths[1:] - wavelengths[:-1] - 1):
      # The wavelengths don't increment by 1nm per reflectance measurement
      raise FileFormatError('Reflectance measurements should exist for every nm')

    wavelengths = wavelengths / 1000
    stacked = []
    for col in df.columns[1:]:
      reflectance = df[col]
      data = np.stack((reflectance, wavelengths), 0)
      stacked.append(data)
    stacked = np.asarray(stacked)

    if stacked.dtype == np.object:
      raise FileFormatError('Each observation in the CSV should be a column of numerical values')

    wl_lower = int(stacked[0, 1, 0] * 1000)
    wl_upper = int(stacked[0, 1, -1] * 1000)
    if stacked.shape[0] > MAX_NUM_SAMPLES:
      stacked = stacked[:MAX_NUM_SAMPLES]
      names = names[:MAX_NUM_SAMPLES]
      truncated = True

    # Handle jump correction (if jumps provided)
    stacked = jump_correction(stacked, jumps)

    # Split into batches on the way out
    num_batches = np.ceil(stacked.shape[0] / MAX_BATCH_SIZE)
    batched = np.array_split(stacked, num_batches, 0)
  except FileFormatError as e:
    # Propagate if we threw the exception
    raise e
  except Exception as e:
    print(e)
    # Catch and re-throw all others
    raise FileFormatError('An error occurred while parsing the CSV file - ensure the format matches the example')

  return batched, names, wl_lower, wl_upper, truncated


def _pass_through_model(model, inputs):
  # Normalise
  inputs[:, 0] = (inputs[:, 0] - STATS['inputs']['mean']) / STATS['inputs']['std']

  # Predict
  raw_preds = model.forward(inputs)

  # Denormalise
  raw_preds = raw_preds * STATS['traits']['std'][None] + STATS['traits']['mean'][None]

  return raw_preds


def make_app(name):
  app = Flask(name)

  def _json_response(data, code=200):
    response = Response(response=json.dumps(data, indent=2),
                        status=code,
                        mimetype='application/json')
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "*")
    response.headers.add("Access-Control-Allow-Methods", "*")
    return response, code

  def _http_error(msg, code):
    return _json_response({
      'error': msg,
      'error_code': code
    }, code)

  def _predict(args, files):
    if 'model' not in args:
      raise RequestError('Must provide a model name like /predict?model=cnn')

    jumps = None
    if 'jumps' in args:
      try:
        jumps = [int(j.strip()) for j in args['jumps'].split(',')]
      except:
        jumps = None

    if 'reflectance' not in files:
      raise RequestError('Must provide a file named \'reflectance\'')
    if os.path.splitext(files['reflectance'].filename)[1].lower() != '.csv':
      raise RequestError('Provided file should have an extension of \'csv\'')

    input_data, names, wl_lower, wl_upper, truncated = _parse_reflectance_data(files['reflectance'], jumps=jumps)
    model = _build_model(args['model'], wl_lower, wl_upper)

    raw_preds = []
    for batch in input_data:
      preds = _pass_through_model(model, batch).tolist()
      raw_preds.extend(preds)

    return raw_preds, names, truncated

  @app.route('/predict', methods=['POST'])
  def post_prediction():
    try:
      predictions, names, truncated = _predict(request.args, request.files)
      response_code = 202 if truncated else 200
      return _json_response(_format_response(predictions, names), response_code)
    except HTTPError as e:
      print("ERROR:", e, flush=True)
      return _http_error(e.msg, e.code)
    except Exception as e:
      print("Error:", e, flush=True)
      return _http_error('An internal error occurred during processing', 500)

  @app.route('/test', methods=['GET'])
  def get_ui():
    out = send_from_directory(BASE_PATH, 'dev_server_interface.html')
    return out

  return app


app = make_app(__name__)
if __name__ == '__main__':
  app.run(host='0.0.0.0', debug=DEV_MODE)
