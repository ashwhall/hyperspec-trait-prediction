"""Process a generic csv data file and store it in `/data/processed`"""
import os
import pandas as pd

import constants


DATASET_VERSION = 'v1'
INPUT_FILENAME = 'Mtrx-LMA-Narea-SPAD_forZH.csv'

in_path = os.path.join(constants.RAW_DIR, DATASET_VERSION, INPUT_FILENAME)
out_path = os.path.join(constants.DATAFRAME_DIR, f'dataset_{DATASET_VERSION}.csv')


def read_file(path):
  df = pd.read_csv(path)
  for trait in constants.TRAITS:
    if trait not in df.columns:
      print("{:<12}: not found".format(trait))
    else:
      print("{:<12}: found".format(trait))

  df = constants.apply_df_filter(df)

  return df


df = read_file(in_path)

# Add the derived trait
if ("Vcmax25" in df.columns and "Narea_O" in df.columns) and 'Vcmax25_Narea_O' not in df.columns:
  df["Vcmax25_Narea_O"] = df["Vcmax25"] / df["Narea_O"]


print(df)
print("Saving...")
df.to_csv(out_path, index=False, header=True)
print("Written to", out_path)
