import pandas as pd
from pathlib import Path

MODEL_FOLDER = "models"

# save to bucket


def save_stuff(model,name):
    try:
        Path(MODEL_FOLDER).mkdir(exist_ok=True, parents=True)
    except Exception as ex:
        pass
    name = name + ".pkl"
    pd.to_pickle(model, f"{MODEL_FOLDER}/{name}.pkl")


def load_stuff(name):
    name = name + ".pkl"
    path = Path(f"{MODEL_FOLDER}/{name}.pkl")
    if path.is_file():
        model = pd.read_pickle(f"{MODEL_FOLDER}/{name}.pkl")
    else:
        model = False
    return model



