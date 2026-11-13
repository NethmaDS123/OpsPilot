"""CLI: ingest policy markdown into PostgreSQL (pgvector) or in-memory store."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.core.config import get_settings
from app.knowledge.embeddings import build_embedding_client
from app.knowledge.ingestion import ingest_policy_directory
from app.knowledge.store import build_vector_store


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest policy markdown into the vector store.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/policies"),
        help="Root directory containing company policy folders.",
    )
    parser.add_argument(
        "--include-superseded",
        action="store_true",
        help="Store superseded policies (retrieval still filters them out).",
    )
    args = parser.parse_args()

    settings = get_settings()
    store = build_vector_store(settings.database_url)
    embedder = build_embedding_client()
    totals = ingest_policy_directory(
        args.root,
        store=store,
        embedder=embedder,
        include_superseded=args.include_superseded,
    )
    backend = "postgres" if settings.database_url else "in-memory"
    print(
        f"Ingested {totals['files']} files / {totals['chunks']} chunks via {backend} backend."
    )


if __name__ == "__main__":
    main()
