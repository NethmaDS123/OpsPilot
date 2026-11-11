from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_classify_requires_llm_configuration():
    """Without OPENAI_API_KEY, the LLM endpoint returns 503."""
    response = client.post(
        "/tickets/classify",
        json={
            "ticket_id": "TICKET-001",
            "message": "I was charged twice for my order",
        },
    )
    assert response.status_code == 503
