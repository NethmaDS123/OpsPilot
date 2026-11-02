from typing import Literal

from pydantic import BaseModel, Field

TicketCategory = Literal[
    "billing",
    "refund",
    "technical",
    "delivery",
    "account",
    "other",
]

TicketPriority = Literal[
    "low",
    "normal",
    "high",
    "urgent",
]


class TicketRequest(BaseModel):
    customer_id: str
    message: str = Field(
        min_length=1,
        max_length=5000,
    )
    order_id: str | None = None


class TicketClassification(BaseModel):
    category: TicketCategory
    priority: TicketPriority
    summary: str = Field(
        min_length=1,
        max_length=300,
    )
    requires_human: bool


class TicketResponse(BaseModel):
    ticket_id: str
    classification: TicketClassification
