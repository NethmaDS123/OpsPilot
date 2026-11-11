"""Shared pytest fixtures: fake LLM client and FastAPI dependency overrides."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.classification import LLMClassificationResult
from app.services.llm_client import (
    ClassificationResponse,
    TicketClassifierClient,
    TokenUsage,
)
from app.services.safety import normalize_message


class FakeClassifierClient(TicketClassifierClient):
    """Deterministic LLM stand-in for tests (no network)."""

    def classify(self, message: str) -> ClassificationResponse:
        text = normalize_message(message).lower()

        if "ignore previous instructions" in text or "reveal your system prompt" in text:
            return ClassificationResponse(
                result=LLMClassificationResult(
                    category="other",
                    priority="high",
                    confidence=0.2,
                    requires_human_review=True,
                    justification="Suspicious instruction override attempt in ticket.",
                ),
                usage=TokenUsage(input_tokens=120, output_tokens=40, total_tokens=160),
            )

        if text in {"?", "help", "help!!!"} or len(text) < 12:
            return ClassificationResponse(
                result=LLMClassificationResult(
                    category="other",
                    priority="normal",
                    confidence=0.35,
                    requires_human_review=True,
                    justification="Message is too vague to classify confidently.",
                ),
                usage=TokenUsage(input_tokens=90, output_tokens=35, total_tokens=125),
            )

        rules: list[tuple[tuple[str, ...], LLMClassificationResult]] = [
            (
                ("charged twice", "duplicate", "debited again", "billing"),
                LLMClassificationResult(
                    category="billing",
                    priority="high",
                    confidence=0.93,
                    requires_human_review=True,
                    justification="Customer reports a duplicate or incorrect charge.",
                ),
            ),
            (
                ("refund", "money back", "return my money"),
                LLMClassificationResult(
                    category="refund",
                    priority="normal",
                    confidence=0.91,
                    requires_human_review=False,
                    justification="Customer is requesting a refund.",
                ),
            ),
            (
                (
                    "hasn't arrived",
                    "tracking",
                    "package",
                    "delivery",
                    "shipped",
                    "replacement",
                    "damaged",
                ),
                LLMClassificationResult(
                    category="delivery",
                    priority="high",
                    confidence=0.94,
                    requires_human_review=False,
                    justification="Customer reports a shipping or delivery issue.",
                ),
            ),
            (
                ("crashing", "error message", "not working", "broken"),
                LLMClassificationResult(
                    category="technical",
                    priority="normal",
                    confidence=0.92,
                    requires_human_review=False,
                    justification="Customer describes a product or software defect.",
                ),
            ),
            (
                ("log in", "password", "hacked", "compromised", "account"),
                LLMClassificationResult(
                    category="account",
                    priority="urgent",
                    confidence=0.95,
                    requires_human_review=True,
                    justification="Customer reports account access or security concern.",
                ),
            ),
            (
                ("store hours", "general question", "newsletter"),
                LLMClassificationResult(
                    category="other",
                    priority="low",
                    confidence=0.88,
                    requires_human_review=False,
                    justification="General inquiry not tied to a specific issue type.",
                ),
            ),
        ]

        for keywords, result in rules:
            if any(k in text for k in keywords):
                return ClassificationResponse(
                    result=result,
                    usage=TokenUsage(
                        input_tokens=100,
                        output_tokens=38,
                        total_tokens=138,
                    ),
                )

        return ClassificationResponse(
            result=LLMClassificationResult(
                category="other",
                priority="normal",
                confidence=0.55,
                requires_human_review=True,
                justification="Could not map message to a single category with confidence.",
            ),
            usage=TokenUsage(input_tokens=80, output_tokens=30, total_tokens=110),
        )


class BrokenOutputClient(TicketClassifierClient):
    def classify(self, message: str) -> ClassificationResponse:
        from app.schemas.classification import MalformedModelOutputError

        raise MalformedModelOutputError("Simulated empty model output.")


class TimeoutClient(TicketClassifierClient):
    def classify(self, message: str) -> ClassificationResponse:
        from app.schemas.classification import ClassificationProviderError

        raise ClassificationProviderError("LLM request timed out.")


@pytest.fixture
def fake_llm_client() -> FakeClassifierClient:
    return FakeClassifierClient()


@pytest.fixture
def api_client(fake_llm_client: FakeClassifierClient) -> TestClient:
    from app.api import tickets as tickets_api

    # Swap real OpenAI client for deterministic keyword-based fake responses.
    app.dependency_overrides[tickets_api.get_classifier_client] = lambda: fake_llm_client
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()
