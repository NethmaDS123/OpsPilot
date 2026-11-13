"""Semantic policy retrieval with mandatory tenant isolation."""

from __future__ import annotations

import re
import time
from app.knowledge.embeddings import EmbeddingClient
from app.knowledge.store import PolicyVectorStore
from app.schemas.knowledge import PolicyChunkRecord, RetrievalResult

_COMPANY_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,62}$")


def validate_company_id(company_id: str) -> str:
    cleaned = company_id.strip().lower()
    if not _COMPANY_ID_RE.fullmatch(cleaned):
        raise ValueError("Invalid company_id.")
    return cleaned


async def retrieve_relevant_policies(
    query: str,
    company_id: str,
    top_k: int = 5,
    *,
    store: PolicyVectorStore,
    embedder: EmbeddingClient,
) -> RetrievalResult:
    """Embed the query and search pgvector scoped to the authenticated company."""
    tenant = validate_company_id(company_id)
    cleaned_query = query.strip()
    if not cleaned_query:
        raise ValueError("Query must not be empty.")

    vector = embedder.embed([cleaned_query])[0]
    chunks = store.search(
        company_id=tenant,
        query_embedding=vector,
        top_k=top_k,
        include_superseded=False,
    )
    return RetrievalResult(query=cleaned_query, company_id=tenant, chunks=chunks)


class TimedRetriever:
    def __init__(
        self,
        store: PolicyVectorStore,
        embedder: EmbeddingClient,
    ) -> None:
        self._store = store
        self._embedder = embedder

    async def retrieve(
        self,
        query: str,
        company_id: str,
        top_k: int = 5,
    ) -> tuple[RetrievalResult, float]:
        start = time.perf_counter()
        result = await retrieve_relevant_policies(
            query,
            company_id,
            top_k=top_k,
            store=self._store,
            embedder=self._embedder,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        return result, latency_ms


def unique_document_ids(chunks: list[PolicyChunkRecord]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for chunk in chunks:
        if chunk.document_id not in seen:
            seen.add(chunk.document_id)
            ordered.append(chunk.document_id)
    return ordered
