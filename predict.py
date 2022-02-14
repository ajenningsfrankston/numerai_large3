

import pandas as pd

from numerapi import NumerAPI

from pathlib import Path

from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest, f_regression

import pyarrow.parquet as pq


TOURNAMENT_NAME= "first_large"
ERA_COL = "era"
TARGET_COL = "target_nomi_20"
DATA_TYPE_COL = "data_type"
EXAMPLE_PREDS_COL = "example_preds"
PREDICTION_NAME = "prediction"

MODEL_FOLDER = "models"


napi = NumerAPI()

# utilities

from utils import load_stuff, save_stuff

def score(df):
    return df[[TARGET_COL, PREDICTION_NAME]].corr(method="spearman")[TARGET_COL][PREDICTION_NAME]


current_round = napi.get_current_round(tournament=8)  # tournament 8 is the primary Numerai Tournament

print('Downloading dataset files...')

napi.download_dataset("numerai_tournament_data.parquet", f"tournament_data_{current_round}.parquet")
napi.download_dataset("numerai_validation_data.parquet", f"validation_data.parquet")

validation_data = pq.read_table('validation_data.parquet').to_pandas()
tournament_data = pq.read_table(f"tournament_data_{current_round}.parquet").to_pandas()

napi.download_dataset("example_validation_predictions.parquet", "example_validation_predictions.parquet")

model_name = f"model_target"
print(f"Checking for existing model '{model_name}'")
model = load_stuff(model_name)
selected_features = load_stuff('features')
if not model:
    print(f"model not found")

model_expected_features = selected_features

read_columns = model_expected_features + [ERA_COL, DATA_TYPE_COL, TARGET_COL]

#validation_data = pd.read_csv('validation_data.csv', usecols=read_columns)
validation_data[PREDICTION_NAME] = model.predict(validation_data[model_expected_features])

validation_preds = pq.read_table("example_validation_predictions.parquet").to_pandas()
validation_data[EXAMPLE_PREDS_COL] = validation_preds["prediction"]

validation_correlations = validation_data.groupby("era").apply(score)
print(f"On validation the correlation has mean {validation_correlations.mean()} and std {validation_correlations.std()}")


#tournament_data = pd.read_csv(f'tournament_data_{current_round}.csv')

print('Predicting on tournament data')
tournament_data[PREDICTION_NAME] = model.predict(tournament_data[model_expected_features])

tournament_data[PREDICTION_NAME].to_csv("predictions.csv")


id = "OML65REYFDPC5O7N22XCRP44BG2M74XH"
key = "YSTL455VERL7WZ4D7OQ6XEYEQN2MRCCICBMILNFP3DUZC4MSAS2WSH2MV7ED6WB3"

napi = NumerAPI(public_id=id,secret_key=key)

# print('uploading')

path =  f"tournament_predictions_{current_round}.csv"

path =  "predictions.csv"

print('uploading')
submission_string = napi.upload_predictions(file_path=path,version=2)
print(submission_string)

#tidy up by removing per tournament files

os.remove(f"tournament_data_{current_round}.parquet")








