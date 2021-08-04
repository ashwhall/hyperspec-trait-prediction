"""Merges two dataset CSVs into one. Pretty much specifically for the v1 dataset"""
import os
import pandas as pd

import constants


DATASET_VERSION = 'v1'
INPUT_FILENAMES = [
  'Mtrx-LMA-Narea-SPAD_forZH.csv',
  'Out_Mtrx_Vcmax-Vc25-J_forZH.csv'
]

file_paths = [
  os.path.join(constants.RAW_DIR, DATASET_VERSION, f)
  for f in INPUT_FILENAMES
]
out_path = os.path.join(constants.DATAFRAME_DIR, f'dataset_{DATASET_VERSION}.csv')

def read_file(path, dataset_num):
  df = pd.read_csv(path)
  df['col_index'] = range(0, len(df))
  df = constants.apply_df_filter(df)
  df = df[df['Wave_350'].notnull()]

  if (dataset_num == 1):
    df = df[df['Exp'] != 'CB_Mex']

  return df


collated = []
for dataset_num in range(0, 2):
  file_path = file_paths[dataset_num]
  print(f"Processing {file_path}...")

  file_contents = read_file(file_path, dataset_num)
  print(f"\tshape: {file_contents.shape}")
  collated.append(file_contents)

print("Collating...")
collated = pd.concat(collated, axis=0, ignore_index=True)
# Rename the derived trait
collated['Vcmax25_Narea_O'] = collated['Vcmax25/Narea_O']
print(f"\tshape: {collated.shape}")

print("Saving...")
collated.to_csv(out_path, index=False, header=True)
print("Written to", out_path)
