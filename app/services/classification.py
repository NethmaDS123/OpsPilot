"""Orchestrates LLM classification and applies post-model safety rules."""

from __future__ import annotations

from app.core.config import get_settings
from app.schemas.classification import (
    ClassificationProviderError,
    LLMClassificationResult,
    TicketClassifyRequest,
    TicketClassifyResponse,
)
from app.services.llm_client import ClassificationResponse, TicketClassifierClient
from app.services.safety import (
    detect_prompt_injection,
    detect_sensitive_content,
    normalize_message,
)


def apply_safety_overrides(
    message: str,
    result: LLMClassificationResult,
) -> LLMClassificationResult:
    """Backend overrides: do not trust the model alone for escalation."""
    settings = get_settings()
    requires_review = result.requires_human_review
    confidence = result.confidence

    if detect_prompt_injection(message):
        requires_review = True
        confidence = min(confidence, 0.4)
    if detect_sensitive_content(message):
        requires_review = True
    if confidence < settings.llm_confidence_threshold:
        requires_review = True

    if requires_review == result.requires_human_review and confidence == result.confidence:
        return result

    return result.model_copy(
        update={
            "requires_human_review": requires_review,
            "confidence": confidence,
        }
    )


def classify_ticket_with_llm(
    request: TicketClassifyRequest,
    client: TicketClassifierClient,
) -> tuple[TicketClassifyResponse, ClassificationResponse]:
    message = normalize_message(request.message)
    provider_response = client.classify(message)
    adjusted = apply_safety_overrides(message, provider_response.result)

    response = TicketClassifyResponse(
        ticket_id=request.ticket_id,
        category=adjusted.category,
        priority=adjusted.priority,
        confidence=adjusted.confidence,
        requires_human_review=adjusted.requires_human_review,
        justification=adjusted.justification,
    )
    updated_provider = ClassificationResponse(
        result=adjusted,
        usage=provider_response.usage,
    )
    return response, updated_provider


def classify_ticket_with_llm_safe(
    request: TicketClassifyRequest,
    client: TicketClassifierClient,
) -> TicketClassifyResponse:
    response, _ = classify_ticket_with_llm(request, client)
    return response
