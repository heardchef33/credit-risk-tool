import typing as t

import pandas as pd
from loguru import logger

from src import __version__ as _version
from src.config.core import config
from src.processing.loading import load_pipeline
from src.processing.validation import validate_inputs

pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
_final_pipeline = load_pipeline(file_name=pipeline_file_name)

# The pipeline's ColumnTransformers select columns by name, so every step must
# output pandas DataFrames. sklearn's global `set_config(transform_output=...)`
# is thread-local: it only applies to the thread that called it, so predictions
# made from any other thread (FastAPI's threadpool, test clients, threaded
# servers) would receive numpy arrays and crash. Setting the output format on
# the pipeline object itself makes it hold in every thread.
_final_pipeline.set_output(transform="pandas")


def make_prediction(
    *,
    input_data: t.Union[pd.DataFrame, dict],
) -> dict:
    """Make a prediction using a saved model pipeline."""

    data = pd.DataFrame(input_data)
    validated_data, errors = validate_inputs(input_data=data)
    results = {
        "predictions": None,
        "prediction_probabilities": None,
        "version": _version,
        "errors": errors,
    }

    if not errors:
        features = validated_data[config.model_settings.features]
        predictions = _final_pipeline.predict(X=features)
        prediction_probabilities = _final_pipeline.predict_proba(X=features)

        logger.debug(f"Prediction probabilities: {prediction_probabilities}")

        results = {
            "predictions": predictions.astype(int),  # type: ignore
            "prediction_probabilities": prediction_probabilities,
            "version": _version,
            "errors": errors,
        }

    return results


if __name__ == "__main__":
    example = {
        "int_rate": [10.5],
        "fico_range_high": [750.0],
        "inq_last_6mths": [1.0],
        "open_il_12m": [2.0],
        "acc_open_past_24mths": [3.0],
        "mort_acc": [1.0],
        "num_tl_op_past_12m": [2.0],
        "percent_bc_gt_75": [25.0],
        "sub_grade": ["A2"],
        "term": ["36 months"],
    }

    print(make_prediction(input_data=example))
