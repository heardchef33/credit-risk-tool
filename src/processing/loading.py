import pandas as pd 
from pathlib import Path 
from config.core import DATASET_DIR, TRAINED_MODEL_DIR, config
from sklearn.pipeline import Pipeline
import joblib

def load_dataset(*, file_name: str) -> pd.DataFrame: # the asterik forces file_name to be a keyboard only name
    """Return pandas dataframe of the loaded dataset.""" 

    dataframe = pd.read_csv(f'{DATASET_DIR}/{file_name}')
    return dataframe

def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model

