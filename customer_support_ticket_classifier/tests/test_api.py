from fastapi.testclient import TestClient
from app.main import app


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["models_loaded"] is True


def test_predict_billing_ticket():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "text": "I was charged twice for my subscription"
            },
        )

    assert response.status_code == 200

    data = response.json()
    assert data["category"] == "Billing"
    assert data["priority"] in {"High", "Medium", "Low"}
    assert 0 <= data["confidence"] <= 1


def test_empty_text_returns_422():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"text": ""},
        )

    assert response.status_code == 422


def test_whitespace_text_returns_422():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"text": "   "},
        )

    assert response.status_code == 422