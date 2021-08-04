import os

# 0.7 train, 0.1 validation and 0.2 test
SPLIT_RATIOS = [.7, .1, .2]

DATA_DIR = "/data"
RAW_DIR = os.path.join(DATA_DIR, 'raw')
DATAFRAME_DIR = os.path.join(DATA_DIR, 'processed')
SPLIT_DIR = os.path.join(DATA_DIR, 'splits')
WEIGHTS_DIR = os.path.join(DATA_DIR, 'model_weights')
STAT_DUMP_PATH = os.path.join(DATA_DIR, 'statistics.json')

TRAITS = [
  "LMA_O",
  "Narea_O",
  "SPAD_O",
  "Nmass_O",
  # "Parea_O",
  # "Pmass_O",
  "Vcmax",
  "Vcmax25",
  "J",
  "Photo_O",
  "Cond_O",
  "Vcmax25_Narea_O"
]
# In case we only want to provide predictions for a subset of the traits later on
SERVER_PREDICT_TRAITS = TRAITS
# SERVER_PREDICT_TRAITS = [t for t in SERVER_PREDICT_TRAITS if t not in ['Parea_O', 'Pmass_O']]

def apply_df_filter(df):
  df['col_index'] = range(0, len(df))
  df = df.query('Wave_800>0.35').query('Wave_800<0.6')
  return df
