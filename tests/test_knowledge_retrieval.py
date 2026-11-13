"""Knowledge base ingestion and retrieval tests."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.knowledge.embeddings import DeterministicEmbeddingClient
from app.knowledge.ingestion import ingest_policy_directory
from app.knowledge.loaders import load_policy_markdown
from app.knowledge.retrieval import retrieve_relevant_policies, validate_company_id
from app.knowledge.store import InMemoryPolicyVectorStore


@pytest.fixture
def policy_store() -> InMemoryPolicyVectorStore:
    store = InMemoryPolicyVectorStore()
    embedder = DeterministicEmbeddingClient()
    ingest_policy_directory(
        Path("data/policies"),
        store=store,
        embedder=embedder,
        include_superseded=True,
    )
    return store


@pytest.fixture
def embedder() -> DeterministicEmbeddingClient:
    return DeterministicEmbeddingClient()


def test_retrieve_refund_policy(policy_store, embedder):
    result = asyncio.run(
        retrieve_relevant_policies(
            "refund window standard products full refund eligibility",
            "acme-corp",
            top_k=3,
            store=policy_store,
            embedder=embedder,
        )
    )
    doc_ids = [c.document_id for c in result.chunks]
    assert "acme-refunds-v2" in doc_ids
    assert "acme-refunds-v1" not in doc_ids


def test_tenant_isolation(policy_store, embedder):
    async def _run():
        acme = await retrieve_relevant_policies(
            "refund policy store credit",
            "acme-corp",
            top_k=5,
            store=policy_store,
            embedder=embedder,
        )
        globex = await retrieve_relevant_policies(
            "refund policy store credit",
            "globex-inc",
            top_k=5,
            store=policy_store,
            embedder=embedder,
        )
        return acme, globex

    acme, globex = asyncio.run(_run())
    assert all(c.company_id == "acme-corp" for c in acme.chunks)
    assert all(c.company_id == "globex-inc" for c in globex.chunks)
    assert any(c.document_id == "globex-refunds-v1" for c in globex.chunks)
    assert all(c.document_id != "globex-refunds-v1" for c in acme.chunks)


def test_superseded_policy_filtered_from_search(policy_store, embedder):
    vector = embedder.embed(["automatic $1000 refund without verification"])[0]
    rows = policy_store.search(
        company_id="acme-corp",
        query_embedding=vector,
        top_k=10,
        include_superseded=False,
    )
    assert all(r.document_id != "acme-refunds-v1" for r in rows)


def test_loader_parses_frontmatter():
    doc = load_policy_markdown(Path("data/policies/acme-corp/refunds.md"))
    assert doc.meta.document_id == "acme-refunds-v2"
    assert doc.meta.company_id == "acme-corp"


def test_invalid_company_id_rejected():
    with pytest.raises(ValueError):
        validate_company_id("../other-tenant")
