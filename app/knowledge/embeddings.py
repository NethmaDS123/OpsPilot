"""Embedding providers: OpenAI for production, deterministic vectors for tests."""

from __future__ import annotations

import hashlib
import math
from typing import Protocol

import numpy as np
from openai import OpenAI

from app.core.config import get_settings

EMBEDDING_DIM = 1536
EMBEDDING_MODEL = "text-embedding-3-small"


class EmbeddingClient(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class OpenAIEmbeddingClient:
    def __init__(self, api_key: str | None = None, model: str = EMBEDDING_MODEL) -> None:
        settings = get_settings()
        key = api_key if api_key is not None else settings.openai_api_key
        if not key:
            raise ValueError("OPENAI_API_KEY is required for embeddings.")
        self._client = OpenAI(api_key=key, timeout=settings.llm_request_timeout)
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self._model, input=texts)
        ordered = sorted(response.data, key=lambda row: row.index)
        return [row.embedding for row in ordered]


class DeterministicEmbeddingClient:
    """Feature-hashed bag-of-words vectors for offline retrieval benchmarks."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [_hashed_bow_vector(text) for text in texts]


def _tokenize(text: str) -> list[str]:
    cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text.lower())
    return [tok for tok in cleaned.split() if len(tok) > 2]


def _hashed_bow_vector(text: str) -> list[float]:
    vec = np.zeros(EMBEDDING_DIM, dtype=np.float64)
    for token in _tokenize(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % EMBEDDING_DIM
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(float(np.dot(vec, vec)))
    if norm == 0:
        return [0.0] * EMBEDDING_DIM
    return (vec / norm).tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Embedding dimensions must match.")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def build_embedding_client() -> EmbeddingClient:
    settings = get_settings()
    if settings.openai_api_key:
        return OpenAIEmbeddingClient()
    return DeterministicEmbeddingClient()
