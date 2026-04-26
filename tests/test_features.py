import pytest
import pandas as pd 
from unittest.mock import patch
from src.config.core import config
from src.processing.loading import load_dataset, load_pipeline, remove_old_pipelines
from src.processing.validation import validate_inputs
from typing import Optional

@pytest.fixture
def sample_data(): 

    data = {
        'loan_status': [
            'Fully Paid',
            'Current',
            'In Grace Period',
            'Does not meet the credit policy. Status:Fully Paid',
            'Does not meet the credit policy. Status:Charged Off',
            'Late (16-30 days)',
            'Late (31-120 days)',
            'Default',
            'Charged Off',
            None
        ],
        'term': [' 36 months', ' 36 months', ' 60 months', ' 60 months', ' 60 months', ' 36 months', ' 36 months', ' 60 months', ' 36 months', ' 60 months']
    }

    return pd.DataFrame(data)

# testing loading
@patch('pandas.read_csv')
@patch('src.config.core.DATASET_DIR', 'mock_dir')
def test_load_dataset(mock_read_csv, sample_data): 

    mock_read_csv.return_value = sample_data

    df_result = load_dataset(file_name='test.csv')

    assert isinstance(df_result, pd.DataFrame) == True
    assert len(df_result) == 5
    assert df_result.iloc[0]['term'] == '36 months'

@patch('joblib.load')
@patch('src.config.core.TRAINED_MODEL_DIR', 'mock_dir')
def test_load_pipeline(mock_read_pickle):
    
    mock_read_pickle.return_value = {"key": "mocked_data"}

    trained_model_result = load_pipeline(file_name='test.pkl')

    assert isinstance(trained_model_result, object) == True

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
    
    remaining_files = [file.name for file in model_dir.iterdir()]

    assert len(remaining_files) == 2
    assert "xgboost_model_output_v1.pkl" in remaining_files
    assert "__init__.py" in remaining_files
    assert "xgboost_model_output_v0.pkl" not in remaining_files
    assert "model.txt" not in remaining_files

# testing validation

def test_validate_inputs(sample_input_data):
    
    validated_data_test, errors_test = validate_inputs(input_data=sample_input_data)

    assert isinstance(validated_data_test, pd.DataFrame)
    assert isinstance(errors_test, Optional[dict])
