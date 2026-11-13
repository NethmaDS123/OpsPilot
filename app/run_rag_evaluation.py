"""Evaluate retrieval and RAG: Precision@3, Recall@3, MRR, latency, and cost estimates."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from statistics import mean

from app.evaluation.rag_eval_dataset import RAG_EVAL_CASES
from app.knowledge.embeddings import DeterministicEmbeddingClient
from app.knowledge.ingestion import ingest_policy_directory
from app.knowledge.rag import HeuristicRAGClient, answer_support_question
from app.knowledge.retrieval import retrieve_relevant_policies, unique_document_ids
from app.knowledge.store import InMemoryPolicyVectorStore


def _precision_at_k(retrieved: list[str], expected: str, k: int = 3) -> float:
    if not expected:
        return 0.0
    top = retrieved[:k]
    hits = sum(1 for doc_id in top if doc_id == expected)
    return hits / k


def _recall_at_k(retrieved: list[str], expected: str, k: int = 3) -> float:
    if not expected:
        return 0.0
    top = retrieved[:k]
    return 1.0 if expected in top else 0.0


def _mrr(retrieved: list[str], expected: str) -> float:
    if not expected:
        return 0.0
    for idx, doc_id in enumerate(retrieved, start=1):
        if doc_id == expected:
            return 1.0 / idx
    return 0.0


async def _run() -> dict:
    store = InMemoryPolicyVectorStore()
    embedder = DeterministicEmbeddingClient()
    rag_client = HeuristicRAGClient()
    root = Path("data/policies")
    ingest_policy_directory(root, store=store, embedder=embedder, include_superseded=True)

    rows: list[dict] = []
    for case in RAG_EVAL_CASES:
        retrieval = await retrieve_relevant_policies(
            case.question,
            case.company_id,
            top_k=5,
            store=store,
            embedder=embedder,
        )
        ranked_docs = unique_document_ids(retrieval.chunks)
        answer = await answer_support_question(
            case.question,
            case.company_id,
            store=store,
            embedder=embedder,
            rag_client=rag_client,
        )

        hit_top3 = case.expected_document_id in ranked_docs[:3]
        answer_ok = case.expect_abstain or (
            not answer.abstain and case.expected_document_id in {s.document_id for s in answer.sources}
        )
        if case.expect_abstain:
            answer_ok = answer.abstain

        rows.append(
            {
                "question": case.question,
                "company_id": case.company_id,
                "expected_document_id": case.expected_document_id,
                "retrieved_top3": ranked_docs[:3],
                "precision_at_3": _precision_at_k(ranked_docs, case.expected_document_id),
                "recall_at_3": _recall_at_k(ranked_docs, case.expected_document_id),
                "mrr": _mrr(ranked_docs, case.expected_document_id),
                "hit_top3": hit_top3,
                "answer_ok": answer_ok,
                "unsupported_claim": answer.unsupported_claim_warning,
                "abstain": answer.abstain,
                "retrieval_latency_ms": round(answer.retrieval_latency_ms, 2),
                "llm_latency_ms": round(answer.llm_latency_ms, 2),
                "estimated_cost_usd": answer.estimated_cost_usd,
            }
        )

    summary = {
        "cases": len(rows),
        "precision_at_3_mean": round(mean(r["precision_at_3"] for r in rows), 4),
        "recall_at_3_mean": round(mean(r["recall_at_3"] for r in rows), 4),
        "mrr_mean": round(mean(r["mrr"] for r in rows), 4),
        "answer_correctness_rate": round(mean(1.0 if r["answer_ok"] else 0.0 for r in rows), 4),
        "unsupported_claims": sum(1 for r in rows if r["unsupported_claim"]),
        "mean_retrieval_latency_ms": round(mean(r["retrieval_latency_ms"] for r in rows), 2),
        "mean_llm_latency_ms": round(mean(r["llm_latency_ms"] for r in rows), 2),
        "mean_estimated_cost_usd": round(
            mean(r["estimated_cost_usd"] or 0.0 for r in rows), 6
        ),
    }
    return {"summary": summary, "rows": rows}


def main() -> None:
    report = asyncio.run(_run())
    print(json.dumps(report["summary"], indent=2))
    out_path = Path("app/artifacts/rag_evaluation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote detailed report to {out_path}")


if __name__ == "__main__":
    main()
