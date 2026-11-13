"""Support knowledge API: semantic retrieval and RAG answers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from app.core.config import get_settings
from app.knowledge.embeddings import EmbeddingClient, build_embedding_client
from app.knowledge.ingestion import ingest_policy_directory
from app.knowledge.rag import RAGAnswerClient, answer_support_question, build_rag_client
from app.knowledge.retrieval import retrieve_relevant_policies
from app.knowledge.store import InMemoryPolicyVectorStore, PolicyVectorStore, build_vector_store
from app.schemas.knowledge import RetrievalResult
from app.schemas.support import PolicySearchRequest, SupportQuestionRequest, SupportQuestionResponse

router = APIRouter(prefix="/support", tags=["support"])


@lru_cache
def _bootstrap_memory_store() -> PolicyVectorStore:
    store = InMemoryPolicyVectorStore()
    embedder = build_embedding_client()
    ingest_policy_directory(
        Path("data/policies"),
        store=store,
        embedder=embedder,
        include_superseded=True,
    )
    return store


def get_policy_store() -> PolicyVectorStore:
    settings = get_settings()
    if settings.database_url:
        return build_vector_store(settings.database_url)
    return _bootstrap_memory_store()


def get_embedder() -> EmbeddingClient:
    return build_embedding_client()


def get_rag_client() -> RAGAnswerClient:
    return build_rag_client()


@router.post("/search", response_model=RetrievalResult)
async def search_policies(
    body: PolicySearchRequest,
    store: PolicyVectorStore = Depends(get_policy_store),
    embedder: EmbeddingClient = Depends(get_embedder),
) -> RetrievalResult:
    try:
        return await retrieve_relevant_policies(
            body.query,
            body.company_id,
            top_k=body.top_k,
            store=store,
            embedder=embedder,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/answer", response_model=SupportQuestionResponse)
async def answer_question(
    body: SupportQuestionRequest,
    store: PolicyVectorStore = Depends(get_policy_store),
    embedder: EmbeddingClient = Depends(get_embedder),
    rag_client: RAGAnswerClient = Depends(get_rag_client),
) -> SupportQuestionResponse:
    try:
        result = await answer_support_question(
            body.question,
            body.company_id,
            store=store,
            embedder=embedder,
            rag_client=rag_client,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return SupportQuestionResponse.model_validate(result.model_dump())
