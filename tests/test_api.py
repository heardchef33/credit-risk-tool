import pytest
from fastapi.testclient import TestClient

from src.api.main import app

PREDICT_URL = "/api/v1/predict"


@pytest.fixture(scope="module")
def client():
    # TestClient runs the app in a different thread from the test, so these
    # tests also guard against the thread-local sklearn config regression.
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):

    response = client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "Loan Default Prediction API"
    assert body["model_version"]


def test_predict_valid_request(client, valid_request_row):

    response = client.post(PREDICT_URL, json={"inputs": [valid_request_row]})

    assert response.status_code == 200
    body = response.json()
    assert body["errors"] is None
    assert body["predictions"] == [0]
    assert isinstance(body["predictions"][0], int)
    assert sum(body["prediction_probabilities"][0]) == pytest.approx(1)


def test_predict_batch(client, valid_request_row):

    second = {**valid_request_row, "sub_grade": "G5", "term": "60 months"}
    response = client.post(PREDICT_URL, json={"inputs": [valid_request_row, second]})

    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2


@pytest.mark.parametrize(
    "field, value",
    [
        ("sub_grade", "A"),
        ("term", "48 months"),
        ("fico_range_high", -5),
        ("fico_range_high", 6.0),
        ("int_rate", "high"),
    ],
)
def test_predict_rejects_invalid_values_with_422(client, valid_request_row, field, value):

    response = client.post(PREDICT_URL, json={"inputs": [{**valid_request_row, field: value}]})

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][-1] == field


def test_predict_rejects_missing_field(client, valid_request_row):

    row = {k: v for k, v in valid_request_row.items() if k != "mort_acc"}
    response = client.post(PREDICT_URL, json={"inputs": [row]})

    assert response.status_code == 422


def test_predict_rejects_unknown_field(client, valid_request_row):

    response = client.post(PREDICT_URL, json={"inputs": [{**valid_request_row, "subgrade": "A2"}]})

    assert response.status_code == 422


def test_predict_rejects_empty_inputs(client):

    response = client.post(PREDICT_URL, json={"inputs": []})

    assert response.status_code == 422


@pytest.mark.parametrize("field", ["fico_range_high", "term", "sub_grade"])
def test_predict_imputes_null_values(client, valid_request_row, field):

    response = client.post(PREDICT_URL, json={"inputs": [{**valid_request_row, field: None}]})

    assert response.status_code == 200
    assert response.json()["errors"] is None
