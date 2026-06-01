from fastapi.testclient import TestClient

from app.main import app


def test_demo_feedback_endpoint_returns_feedback():
    client = TestClient(app)

    response = client.post("/api/demo-feedback", json={"target_tone": 2})

    assert response.status_code == 200
    payload = response.json()
    assert payload["target_tone"] == 2
    assert 1 <= payload["predicted_tone"] <= 4
    assert payload["feedback"]
    assert payload["user_contour"]
    assert payload["target_contour"]
