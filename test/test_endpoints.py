from fastapi.testclient import TestClient
from api.app import app


client = TestClient(app)


def test_can_import_app():
    assert app is not None


def test_predict_returns_valid_json_response():
    response = client.post("/predict", json={"text": "I love this product"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")

    data = response.json()
    assert isinstance(data, dict)
    assert "prediction" in data
    assert isinstance(data["prediction"], str)
    assert data["prediction"] in {"negative", "neutral", "positive"}


def test_predict_positive_case():
    response = client.post("/predict", json={"text": "I love this product"})

    assert response.status_code == 200
    assert response.json() == {"prediction": "positive"}


def test_predict_negative_case():
    response = client.post(
        "/predict",
        json={"text": "This was a terrible experience and I want a refund"},
    )

    assert response.status_code == 200
    assert response.json() == {"prediction": "negative"}


def test_predict_neutral_case():
    response = client.post(
        "/predict",
        json={"text": "The package arrived yesterday and I opened it"},
    )

    assert response.status_code == 200
    assert response.json() == {"prediction": "neutral"}


def test_predict_sentiment_invalid_input_format():
    response = client.post("/predict", json={})
    assert response.status_code == 422
    assert "Field required" in response.text
    assert response.headers["content-type"] == "application/json"
