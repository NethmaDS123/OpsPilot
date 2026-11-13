"""Ingest policy markdown: load, chunk, embed, and persist with metadata."""

from __future__ import annotations

from pathlib import Path

from app.knowledge.chunking import chunk_markdown
from app.knowledge.embeddings import EmbeddingClient
from app.knowledge.loaders import iter_policy_files, load_policy_markdown
from app.knowledge.store import PolicyVectorStore


def ingest_policy_file(
    path: Path,
    *,
    store: PolicyVectorStore,
    embedder: EmbeddingClient,
    include_superseded: bool = True,
) -> int:
    loaded = load_policy_markdown(path)
    if not include_superseded and loaded.meta.status != "active":
        return 0

    store.upsert_document(loaded.meta)
    chunks = chunk_markdown(loaded.body)
    if not chunks:
        return 0

    texts = [
        f"{loaded.meta.title} {loaded.meta.category} {c.content}"
        for c in chunks
    ]
    vectors = embedder.embed(texts)
    payload: list[tuple[int, str, dict, list[float]]] = []
    for chunk, vector in zip(chunks, vectors, strict=True):
        payload.append(
            (
                chunk.index,
                chunk.content,
                {
                    "title": loaded.meta.title,
                    "category": loaded.meta.category,
                    "version": loaded.meta.version,
                    "status": loaded.meta.status,
                    "section": chunk.section,
                    "source_path": loaded.meta.source_path,
                },
                vector,
            )
        )
    store.upsert_chunks(loaded.meta.company_id, loaded.meta.document_id, payload)
    return len(payload)


def ingest_policy_directory(
    root: Path,
    *,
    store: PolicyVectorStore,
    embedder: EmbeddingClient,
    include_superseded: bool = True,
) -> dict[str, int]:
    totals: dict[str, int] = {"files": 0, "chunks": 0}
    for path in iter_policy_files(root):
        chunk_count = ingest_policy_file(
            path,
            store=store,
            embedder=embedder,
            include_superseded=include_superseded,
        )
        if chunk_count:
            totals["files"] += 1
            totals["chunks"] += chunk_count
    return totals
