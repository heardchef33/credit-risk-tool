from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.processing.loading import load_dataset, load_pipeline, remove_old_pipelines
from src.processing.validation import validate_inputs


@pytest.fixture
def sample_data():

    data = {
        "loan_status": [
            "Fully Paid",
            "Current",
            "In Grace Period",
            "Does not meet the credit policy. Status:Fully Paid",
            "Does not meet the credit policy. Status:Charged Off",
            "Late (16-30 days)",
            "Late (31-120 days)",
            "Default",
            "Charged Off",
            None,
        ],
        "term": [" 36 months", " 36 months", " 60 months", " 60 months", " 60 months",
                 " 36 months", " 36 months", " 60 months", " 36 months", " 60 months"],
    }

    return pd.DataFrame(data)


# --- loading -----------------------------------------------------------------

# Patch the names *where they are used* (src.processing.loading), not where they
# are defined: loading.py imported its own reference to DATASET_DIR at import
# time, so patching src.config.core.DATASET_DIR would have no effect.
@patch("src.processing.loading.pd.read_csv")
@patch("src.processing.loading.DATASET_DIR", "mock_dir")
def test_load_dataset(mock_read_csv, sample_data):

    mock_read_csv.return_value = sample_data

    df_result = load_dataset(file_name="test.csv")

    mock_read_csv.assert_called_once_with("mock_dir/test.csv")
    assert isinstance(df_result, pd.DataFrame)
    # Current, In Grace Period, the two policy rows and the null are dropped.
    assert len(df_result) == 5
    # Late and Default loans are relabelled as Charged Off.
    assert df_result["loan_status"].tolist() == ["Fully Paid"] + ["Charged Off"] * 4
    # Leading whitespace in term is stripped.
    assert set(df_result["term"]) == {"36 months", "60 months"}


@patch("src.processing.loading.joblib.load")
def test_load_pipeline(mock_joblib_load, tmp_path):

    sentinel = object()
    mock_joblib_load.return_value = sentinel

    with patch("src.processing.loading.TRAINED_MODEL_DIR", tmp_path):
        trained_model_result = load_pipeline(file_name="test.pkl")

    mock_joblib_load.assert_called_once_with(filename=tmp_path / "test.pkl")
    assert trained_model_result is sentinel


def test_remove_old_pipelines(tmp_path):

    model_dir = tmp_path / "models"
    model_dir.mkdir()

    keep = model_dir / "xgboost_model_output_v1.pkl"
    init = model_dir / "__init__.py"
    delete_0 = model_dir / "xgboost_model_output_v0.pkl"
    delete_1 = model_dir / "model.txt"

    for file in [keep, init, delete_0, delete_1]:
        file.touch()

    with patch("src.processing.loading.TRAINED_MODEL_DIR", model_dir):
        remove_old_pipelines(files_to_keep=["xgboost_model_output_v1.pkl"])

    remaining_files = sorted(file.name for file in model_dir.iterdir())

    assert remaining_files == ["__init__.py", "xgboost_model_output_v1.pkl"]


# --- validation ----------------------------------------------------------------

def test_validate_inputs_valid(sample_input_data):

    validated_data, errors = validate_inputs(input_data=sample_input_data)

    assert errors is None
    assert isinstance(validated_data, pd.DataFrame)
    assert len(validated_data) == 2


@pytest.mark.parametrize(
    "column, bad_value, error_type",
    [
        ("sub_grade", "A", "literal_error"),
        ("sub_grade", "H1", "literal_error"),
        ("term", "48 months", "literal_error"),
        ("fico_range_high", 6.0, "greater_than_equal"),
        ("fico_range_high", 900, "less_than_equal"),
        ("int_rate", -1, "greater_than_equal"),
        ("percent_bc_gt_75", 101, "less_than_equal"),
        ("mort_acc", -2, "greater_than_equal"),
    ],
)
def test_validate_inputs_rejects_invalid_values(sample_input_data, column, bad_value, error_type):

    data = sample_input_data.copy()
    data[column] = data[column].astype(object)
    data.loc[0, column] = bad_value

    _, errors = validate_inputs(input_data=data)

    assert isinstance(errors, list)
    assert errors[0]["type"] == error_type
    assert errors[0]["loc"][-1] == column


def test_validate_inputs_reports_missing_columns(sample_input_data):

    _, errors = validate_inputs(input_data=sample_input_data.drop(columns=["mort_acc", "term"]))

    assert sorted(error["loc"][0] for error in errors) == ["mort_acc", "term"]


def test_validate_inputs_normalises_whitespace_and_case(sample_input_data):

    data = sample_input_data.copy()
    data["term"] = [" 36 months", "60 months "]
    data["sub_grade"] = ["a1", " g1"]

    validated_data, errors = validate_inputs(input_data=data)

    assert errors is None
    assert validated_data["term"].tolist() == ["36 months", "60 months"]
    assert validated_data["sub_grade"].tolist() == ["A1", "G1"]


def test_validate_inputs_turns_nulls_into_nan_for_imputation(sample_input_data):

    data = sample_input_data.copy()
    data["fico_range_high"] = data["fico_range_high"].astype(object)
    data.loc[0, "fico_range_high"] = None
    data.loc[1, "term"] = None

    validated_data, errors = validate_inputs(input_data=data)

    assert errors is None
    # SimpleImputer only recognises np.nan as missing, never None.
    assert np.isnan(validated_data.loc[0, "fico_range_high"])
    term = validated_data.loc[1, "term"]
    assert isinstance(term, float) and np.isnan(term)
