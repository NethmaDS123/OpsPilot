"""Pydantic models for policy ingestion, retrieval, and RAG answers."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field


class PolicyDocumentMeta(BaseModel):
    document_id: str
    title: str
    company_id: str
    category: str
    version: str
    status: Literal["active", "superseded", "draft"] = "active"
    effective_from: date | None = None
    effective_to: date | None = None
    source_path: str


class PolicyChunkRecord(BaseModel):
    chunk_id: str
    company_id: str
    document_id: str
    chunk_index: int
    content: str
    metadata: dict[str, Any]
    similarity: float | None = None


class RetrievalResult(BaseModel):
    query: str
    company_id: str
    chunks: list[PolicyChunkRecord]


class SourceReference(BaseModel):
    document_id: str
    title: str
    chunk_index: int
    category: str


class SupportAnswer(BaseModel):
    question: str
    company_id: str
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    abstain: bool
    abstain_reason: str | None = None
    sources: list[SourceReference] = Field(default_factory=list)
    unsupported_claim_warning: bool = False
    retrieval_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    estimated_cost_usd: float | None = None


class RAGAnswerPayload(BaseModel):
    """Structured LLM output for support answers."""

    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    abstain: bool
    abstain_reason: str | None = None
    cited_document_ids: list[str] = Field(default_factory=list)
