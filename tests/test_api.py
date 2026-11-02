from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_classify_endpoint_success():
    response = client.post(
        "/tickets/classify",
        json={
            "customer_id": "cust_123",
            "message": "I was charged twice for my order",
            "order_id": "ord_456",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert "ticket_id" in body
    assert body["classification"]["category"] == "billing"
    assert body["classification"]["priority"] == "high"
    assert body["classification"]["requires_human"] is True


def test_classify_endpoint_validation_error():
    response = client.post(
        "/tickets/classify",
        json={
            "customer_id": "cust_123",
            "message": "",
        },
    )
    assert response.status_code == 422
