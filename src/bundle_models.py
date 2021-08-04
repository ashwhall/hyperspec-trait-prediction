import os
import pickle
from shutil import copy
import re

import torch

import dataset.constants as constants


RESULTS_DIR = os.path.join('..', 'results')
MODEL_SINGLE = 'ThiSingleTask'
VARIANT_SINGLE = 'base'
MODEL_MULTI = 'ThiMultiTask'
VARIANT_MULTI = 'base'
MODEL_PLS = 'pls'
VARIANT_PLS = 'base'
TRAIN_DS = 'v1'
WEIGHTS_FILE = 'model.pth.tar'
PLS_WEIGHTS_FILE = 'models.bin'

OUTPUT_PATH_SINGLE = os.path.join(constants.WEIGHTS_DIR, 'SingleCNN.bin')
OUTPUT_PATH_MULTI = os.path.join(constants.WEIGHTS_DIR, 'MultiCNN.pth.tar')
OUTPUT_PATH_PLS = os.path.join(constants.WEIGHTS_DIR, 'pls_LOWER-UPPER.bin')


def make_pls_output_path(lower, upper):
  return OUTPUT_PATH_PLS.replace('LOWER', str(lower)).replace('UPPER', str(upper))


def bundle_cnn_single():
  models = {}
  for trait in constants.TRAITS:
    path = os.path.join(RESULTS_DIR,
                        MODEL_SINGLE,
                        f'{VARIANT_SINGLE}_{trait}_{TRAIN_DS}',
                        WEIGHTS_FILE)
    if os.path.isfile(path):
      models[trait] = torch.load(path, map_location=torch.device('cpu'))
    else:
      print("\tWARNING: No model found for trait", trait)
      print("\t\tfull path:", path)

  if len(models) == 0:
    print("\tNothing to bundle!")
  else:
    with open(OUTPUT_PATH_SINGLE, 'wb') as f:
      pickle.dump(models, f)
    print(f"\tWeights saved to \"{OUTPUT_PATH_SINGLE}\"")


def bundle_cnn_multi():
  path = os.path.join(RESULTS_DIR,
                      MODEL_MULTI,
                      f'{VARIANT_MULTI}_{TRAIN_DS}',
                      WEIGHTS_FILE)
  if os.path.isfile(path):
    copy(path, OUTPUT_PATH_MULTI)
    print(f"\tWeights saved to \"{OUTPUT_PATH_MULTI}\"")
  else:
    print("\tWARNING: No CNN-multi model found")
    print("\t\tfull path:", path)


def do_bundle_pls(name, lower, upper):
  path = os.path.join(RESULTS_DIR,
                      MODEL_PLS,
                      name,
                      PLS_WEIGHTS_FILE)
  if os.path.isfile(path):
    out_path = make_pls_output_path(lower, upper)
    copy(path, out_path)
    print(f"\tWeights saved to \"{out_path}\"")
  else:
    print("\tWARNING: No PLS model found")
    print("\t\tfull path:", path)


def bundle_pls():
  do_bundle_pls(f'{VARIANT_PLS}_{TRAIN_DS}', 400, 2400)

  path = os.path.join(RESULTS_DIR,
                      MODEL_PLS)
  for res in os.listdir(path):
    match = re.match('{}_{}_([0-9]*)-([0-9]*)'.format(VARIANT_PLS, TRAIN_DS), res)
    if match:
      s_lower, s_upper = match.group(1), match.group(2)
      do_bundle_pls(res, s_lower, s_upper)


print("Bundling CNN-single...")
bundle_cnn_single()
print("Bundling CNN-multi...")
bundle_cnn_multi()
print("Bundling PLS...")
bundle_pls()
print("Done.")
