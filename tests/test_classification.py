"""Functional tests for the LLM classify route (uses fake client from conftest)."""

import pytest
from fastapi.testclient import TestClient

from app.schemas.classification import TicketClassifyRequest
from app.services.classification import classify_ticket_with_llm_safe
from tests.conftest import BrokenOutputClient, FakeClassifierClient, TimeoutClient


def _post(client: TestClient, ticket_id: str, message: str):
    return client.post(
        "/tickets/classify",
        json={"ticket_id": ticket_id, "message": message},
    )


def test_delivery_damaged_replacement(api_client: TestClient):
    response = _post(
        api_client,
        "TICKET-001",
        "My order arrived damaged and I need a replacement.",
    )
    assert response.status_code == 200
    body = response.json()
    assert body["ticket_id"] == "TICKET-001"
    assert body["category"] == "delivery"
    assert body["confidence"] >= 0.9
    assert body["justification"]


def test_billing_duplicate_charge(api_client: TestClient):
    response = _post(
        api_client,
        "TICKET-002",
        "I was charged twice for the same order.",
    )
    body = response.json()
    assert body["category"] == "billing"
    assert body["requires_human_review"] is True


def test_refund_request(api_client: TestClient):
    response = _post(
        api_client,
        "TICKET-003",
        "Please issue a refund for order ord_88.",
    )
    assert response.json()["category"] == "refund"


def test_technical_crash(api_client: TestClient):
    response = _post(
        api_client,
        "TICKET-004",
        "The mobile app keeps crashing with an error message.",
    )
    assert response.json()["category"] == "technical"


def test_account_login(api_client: TestClient):
    response = _post(
        api_client,
        "TICKET-005",
        "I cannot log in and need to reset my password.",
    )
    body = response.json()
    assert body["category"] == "account"
    assert body["priority"] == "urgent"


def test_general_other(api_client: TestClient):
    response = _post(
        api_client,
        "TICKET-006",
        "Hello, I have a general question about your store hours.",
    )
    body = response.json()
    assert body["category"] == "other"
    assert body["priority"] == "low"


def test_empty_message_rejected(api_client: TestClient):
    assert api_client.post(
        "/tickets/classify",
        json={"ticket_id": "TICKET-007", "message": ""},
    ).status_code == 422


def test_whitespace_only_rejected_via_service():
    client = FakeClassifierClient()
    with pytest.raises(ValueError):
        classify_ticket_with_llm_safe(
            TicketClassifyRequest(ticket_id="T-1", message="   "),
            client,
        )


def test_extremely_long_message_accepted(api_client: TestClient):
    long_text = "My package hasn't arrived. " + ("Still waiting. " * 400)
    long_text = long_text[:5000]
    response = _post(api_client, "TICKET-008", long_text)
    assert response.status_code == 200
    assert response.json()["category"] == "delivery"


def test_overlong_message_rejected(api_client: TestClient):
    response = _post(api_client, "TICKET-009", "x" * 5001)
    assert response.status_code == 422


def test_ml_legacy_endpoint_still_available(api_client: TestClient):
    response = api_client.post(
        "/tickets/classify/ml",
        json={
            "customer_id": "cust_1",
            "message": "I was charged twice for my order",
        },
    )
    assert response.status_code == 200
    assert "classification" in response.json()


def test_provider_timeout_returns_503(api_client: TestClient):
    from app.api import tickets as tickets_api
    from app.main import app

    app.dependency_overrides[tickets_api.get_classifier_client] = lambda: TimeoutClient()
    client = TestClient(app)
    response = _post(client, "TICKET-010", "Any message here for timeout test.")
    app.dependency_overrides.clear()
    assert response.status_code == 503


def test_malformed_model_output_returns_503():
    from app.api import tickets as tickets_api
    from app.main import app

    app.dependency_overrides[tickets_api.get_classifier_client] = (
        lambda: BrokenOutputClient()
    )
    client = TestClient(app)
    response = _post(client, "TICKET-011", "Refund please for damaged item.")
    app.dependency_overrides.clear()
    assert response.status_code == 503
