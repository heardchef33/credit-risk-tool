import pytest 
import numpy as np

from src.predict import make_prediction

# test predict 

def test_make_prediction(sample_input_data):

    expected_first_prediction_value = 0 
    expected_number_predictions = 2 
    
    results = make_prediction(input_data=sample_input_data)

    predictions = results.get("predictions")
    prediction_probabilities = results.get("prediction_probabilities")

    assert len(predictions) == expected_number_predictions
    assert isinstance(predictions, np.ndarray)
    assert isinstance(predictions[0], np.int64)
    assert predictions[0] == expected_first_prediction_value
    assert results.get("errors") is None 
    assert np.sum(prediction_probabilities[0]) == pytest.approx(1)