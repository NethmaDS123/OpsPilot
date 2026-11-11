"""
LLM classification request/response shapes and structured-output model.

Reuses category/priority literals from app.models so all layers stay aligned.
"""

from typing import Literal

from pydantic import BaseModel, Field

from app.models import TicketCategory, TicketPriority


class TicketClassifyRequest(BaseModel):
    ticket_id: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=5000)


class LLMClassificationResult(BaseModel):
    """Structured output returned by the LLM (validated after parsing)."""

    category: TicketCategory
    priority: TicketPriority
    confidence: float = Field(ge=0.0, le=1.0)
    requires_human_review: bool
    justification: str = Field(min_length=1, max_length=500)


class TicketClassifyResponse(BaseModel):
    ticket_id: str
    category: TicketCategory
    priority: TicketPriority
    confidence: float = Field(ge=0.0, le=1.0)
    requires_human_review: bool
    justification: str = Field(min_length=1, max_length=500)


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class ClassificationProviderError(Exception):
    """Raised when the LLM provider fails or returns unusable output."""


class MalformedModelOutputError(ClassificationProviderError):
    """Raised when parsed model output is missing or fails validation."""


LLMCategoriesDoc = Literal[
    "billing",
    "refund",
    "technical",
    "delivery",
    "account",
    "other",
]
