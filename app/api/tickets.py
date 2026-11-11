"""
HTTP layer for ticket classification.

LLM is the default path; /tickets/classify/ml keeps the ml endpoint.
"""

from uuid import uuid4


from fastapi import APIRouter, Depends, HTTPException

from app.classifier import classify_ticket as classify_ticket_ml
from app.models import TicketRequest, TicketResponse
from app.schemas.classification import (
    ClassificationProviderError,
    TicketClassifyRequest,
    TicketClassifyResponse,
)
from app.services.classification import classify_ticket_with_llm_safe
from app.services.llm_client import TicketClassifierClient, build_llm_client

router = APIRouter(tags=["tickets"])


def get_classifier_client() -> TicketClassifierClient:
    # FastAPI Depends() calls this each request; tests override it with a fake client.
    try:
        return build_llm_client()
    except ClassificationProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post(
    "/tickets/classify",
    response_model=TicketClassifyResponse,
)
def classify_ticket_llm(
    body: TicketClassifyRequest,
    client: TicketClassifierClient = Depends(get_classifier_client),
) -> TicketClassifyResponse:
    try:
        return classify_ticket_with_llm_safe(body, client)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ClassificationProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post(
    "/tickets/classify/ml",
    response_model=TicketResponse,
    tags=["tickets", "legacy"],
)
def classify_ticket_traditional_ml(ticket: TicketRequest) -> TicketResponse:
    """TF-IDF + logistic regression pipeline (legacy request schema)."""
    classification = classify_ticket_ml(ticket)
    return TicketResponse(
        ticket_id=str(uuid4()),
        classification=classification,
    )
