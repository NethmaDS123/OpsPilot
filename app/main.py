from uuid import uuid4

from fastapi import FastAPI

from app.models import (
    TicketRequest,
    TicketResponse,
)
from app.classifier import classify_ticket

app = FastAPI(
    title="OpsPilot API",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
    }


@app.post(
    "/tickets/classify",
    response_model=TicketResponse,
)
def classify(
    ticket: TicketRequest,
) -> TicketResponse:
    classification = classify_ticket(ticket)
    return TicketResponse(
        ticket_id=str(uuid4()),
        classification=classification,
    )
