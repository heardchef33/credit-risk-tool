import pandas as pd 
from pathlib import Path 
from src.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config
from sklearn.pipeline import Pipeline
import joblib
import typing as t 
from src import __version__ as _version

def load_dataset(*, file_name: str) -> pd.DataFrame: 
    """Return pandas dataframe of the loaded dataset.""" 

    dataframe = pd.read_csv(f'{DATASET_DIR}/{file_name}')

    dataframe.drop(dataframe.loc[(dataframe['loan_status'] == 'Current') | (dataframe['loan_status'] == 'In Grace Period') | (dataframe['loan_status'] == 'Does not meet the credit policy. Status:Fully Paid') | (dataframe['loan_status'] == 'Does not meet the credit policy. Status:Charged Off')].index, inplace=True)

    dataframe.loc[dataframe['loan_status'] == 'Late (16-30 days)', 'loan_status'] = 'Charged Off'

    dataframe.loc[dataframe['loan_status'] == 'Late (31-120 days)', 'loan_status'] = 'Charged Off'

    dataframe.loc[dataframe['loan_status'] == 'Default', 'loan_status'] = 'Charged Off'

    dataframe = dataframe.dropna(subset=['loan_status'])

    dataframe['term'] = dataframe['term'].str.strip()

    return dataframe

def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model

def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)

def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()

