"""
LLM integration: Protocol for swappable providers, OpenAI Responses implementation.

Tests inject a fake client; production uses OPENAI_API_KEY from .env.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError

from app.core.config import get_settings
from app.prompts.ticket_classifier import SYSTEM_PROMPT
from app.schemas.classification import (
    ClassificationProviderError,
    LLMClassificationResult,
    MalformedModelOutputError,
    TokenUsage,
)


@dataclass
class ClassificationResponse:
    result: LLMClassificationResult
    usage: TokenUsage


class TicketClassifierClient(Protocol):
    def classify(self, message: str) -> ClassificationResponse:
        """Send ticket text to an LLM and return validated structured output."""


class OpenAIResponsesClient:
    """OpenAI Responses API with structured parsing via Pydantic."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        settings = get_settings()
        key = api_key if api_key is not None else settings.openai_api_key
        if not key:
            raise ClassificationProviderError(
                "OPENAI_API_KEY is not set. Add it to .env (never commit the key)."
            )
        self._client = OpenAI(api_key=key, timeout=settings.llm_request_timeout)
        self._model = model or settings.openai_model

    def classify(self, message: str) -> ClassificationResponse:
        try:
            # responses.parse maps provider JSON -> LLMClassificationResult via Pydantic
            response = self._client.responses.parse(
                model=self._model,
                instructions=SYSTEM_PROMPT,
                input=message,
                text_format=LLMClassificationResult,
            )
        except APITimeoutError as exc:
            raise ClassificationProviderError("LLM request timed out.") from exc
        except (APIConnectionError, RateLimitError) as exc:
            raise ClassificationProviderError(
                "LLM provider connection or rate limit error."
            ) from exc
        except Exception as exc:
            raise ClassificationProviderError("LLM provider error.") from exc

        parsed = response.output_parsed
        if parsed is None:
            raise MalformedModelOutputError("Model returned no parsed classification.")

        usage = response.usage
        token_usage = TokenUsage(
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
        )
        return ClassificationResponse(result=parsed, usage=token_usage)


def build_llm_client() -> TicketClassifierClient:
    settings = get_settings()
    if settings.openai_api_key:
        return OpenAIResponsesClient()
    raise ClassificationProviderError(
        "No LLM provider configured. Set OPENAI_API_KEY in .env."
    )
