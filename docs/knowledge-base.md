# OpsPilot Knowledge Base (Week 4)

## Components

| Area | Module |
|------|--------|
| Policy corpus | `data/policies/{company_id}/*.md` |
| Ingestion | `app/knowledge/ingestion.py`, `python -m app.ingest_policies` |
| Embeddings | `app/knowledge/embeddings.py` |
| pgvector store | `app/knowledge/store.py`, `migrations/001_init_pgvector.sql` |
| Retrieval | `app/knowledge/retrieval.py` → `retrieve_relevant_policies` |
| RAG | `app/knowledge/rag.py` → `answer_support_question` |
| HTTP | `POST /support/search`, `POST /support/answer` |
| Evaluation | `app/run_rag_evaluation.py`, `app/evaluation/rag_eval_dataset.py` |

## PostgreSQL setup

```bash
docker compose up -d
export DATABASE_URL=postgresql+psycopg://opspilot:opspilot@localhost:5432/opspilot
pip install -r requirements.txt
python -m app.ingest_policies --include-superseded
```

Without `DATABASE_URL`, the API bootstraps an in-memory vector index from `data/policies/` for local demos and pytest.

## Security controls

- `company_id` validated and enforced in SQL / store search (tenant isolation).
- Superseded and date-expired policies excluded from retrieval (`status = active`, effective dates).
- Prompt-injection heuristics on customer questions (`app/services/safety.py`).
- RAG prompt requires citations; backend flags unsupported `cited_document_ids`.
- Server-side guard abstains on fabricated high-value refund claims.

## Evaluation

```bash
python -m app.run_rag_evaluation
```

Writes `app/artifacts/rag_evaluation.json` with Precision@3, Recall@3, MRR, answer correctness, latency, and estimated cost per case.
