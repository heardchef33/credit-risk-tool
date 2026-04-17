import typing as t 

import numpy as np 
import pandas as pd 

from src import __version__ as _version
from src.config.core import config
from src.processing.loading import load_pipeline
from src.processing.validation import validate_inputs
from sklearn._config import set_config

set_config(transform_output="pandas")

pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
_final_pipeline = load_pipeline(file_name=pipeline_file_name)

def make_prediction(
    *,
    input_data: t.Union[pd.DataFrame, dict],
) -> dict:
    """Make a prediction using a saved model pipeline."""

    data = pd.DataFrame(input_data)
    validated_data, errors = validate_inputs(input_data=data)
    results = {"predictions": None, "version": _version, "errors": errors}

    if not errors:
        predictions = _final_pipeline.predict(
            X=validated_data[config.model_settings.features]
        )
        results = {
            "predictions": predictions,  # type: ignore
            "version": _version,
            "errors": errors,
        }

    return results

if __name__ == "__main__": 
    input = {
        "int_rate" : [2],
        "fico_range_high" : [6.0],
        "inq_last_6mths" : [1.0],
        "open_il_12m" : [1.0],
        "acc_open_past_24mths" : [2.0],
        "mort_acc" : [2],
        "num_tl_op_past_12m" : [3.0],
        "percent_bc_gt_75" : [4.0],
        "sub_grade" : ['A'],
        "term" : ['36 months']
    }

    print(make_prediction(input_data=input))