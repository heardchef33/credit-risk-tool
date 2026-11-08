import threading

import numpy as np
import pytest

from src.predict import make_prediction


def test_make_prediction(sample_input_data):

    expected_first_prediction_value = 0
    expected_number_predictions = 2

    results = make_prediction(input_data=sample_input_data)

    predictions = results.get("predictions")
    prediction_probabilities = results.get("prediction_probabilities")

    assert results.get("errors") is None
    assert len(predictions) == expected_number_predictions
    assert isinstance(predictions, np.ndarray)
    assert np.issubdtype(predictions.dtype, np.integer)
    assert predictions[0] == expected_first_prediction_value
    assert np.sum(prediction_probabilities[0]) == pytest.approx(1)


def test_make_prediction_returns_errors_instead_of_raising(sample_input_data):

    data = sample_input_data.copy()
    data.loc[0, "sub_grade"] = "Z9"

    results = make_prediction(input_data=data)

    assert results["predictions"] is None
    assert results["errors"][0]["loc"][-1] == "sub_grade"


def test_make_prediction_imputes_missing_values(sample_input_data):

    data = sample_input_data.astype(object)
    data.loc[0, ["fico_range_high", "int_rate", "term"]] = None

    results = make_prediction(input_data=data)

    assert results["errors"] is None
    assert len(results["predictions"]) == 2


def test_make_prediction_works_from_another_thread(sample_input_data):
    """Regression test: sklearn's set_config is thread-local, so the pipeline
    must carry its own pandas output setting to work in worker threads."""

    expected = make_prediction(input_data=sample_input_data)
    outcome = {}

    def worker():
        try:
            outcome["results"] = make_prediction(input_data=sample_input_data)
        except Exception as error:  # pragma: no cover - only hit on regression
            outcome["error"] = error

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()

    assert "error" not in outcome, outcome.get("error")
    np.testing.assert_allclose(
        outcome["results"]["prediction_probabilities"], expected["prediction_probabilities"]
    )
