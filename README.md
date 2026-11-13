# OpsPilot

OpsPilot is a **customer-support triage API** that classifies inbound tickets by category and priority, estimates confidence, and flags cases for human review. It demonstrates three production-relevant approaches on the same problem: **keyword rules**, **traditional ML** (TF-IDF + logistic regression), and **LLM classification** with structured outputs and server-side safety controls.

## Features

- **LLM classification** — `POST /tickets/classify` via OpenAI Responses API, Pydantic-validated JSON, post-model safety overrides.
- **ML endpoint** — `POST /tickets/classify/ml` using a trained scikit-learn pipeline on 300+ labeled examples.
- **Rule baseline** — Deterministic keyword classifier for benchmarks and escalation heuristics.
- **Evaluation tooling** — Scripts to compare accuracy, F1, latency, and human-review rates on a shared hold-out split.
- **Test suite** — pytest coverage for rules, ML, LLM routes (with injectable fake LLM), and safety cases.

## Architecture

| Module | Responsibility |
|--------|----------------|
| `app/api/tickets.py` | HTTP routes and dependency injection |
| `app/services/llm_client.py` | Provider abstraction + OpenAI client |
| `app/services/classification.py` | Orchestration and safety overrides |
| `app/services/safety.py` | Input normalization, injection/sensitivity heuristics |
| `app/prompts/ticket_classifier.py` | Versioned system prompt |
| `app/schemas/classification.py` | LLM request/response models |
| `app/rule_classifier.py` | Rule-based classifier |
| `app/ml_classifier.py` | ML category + rule-based priority/escalation |
| `app/train_classifier.py` | Train and persist sklearn pipeline |
| `app/compare_three_classifiers.py` | Rule vs ML vs LLM benchmark |

```
POST /tickets/classify
        │
        ▼
   Pydantic request validation
        │
        ▼
   LLM (structured output) + safety overrides
        │
        ▼
   TicketClassifyResponse
```

Further detail: [`docs/architecture.md`](docs/architecture.md).

## API

### Classify (LLM)

```http
POST /tickets/classify
```

```json
{
  "ticket_id": "TICKET-001",
  "message": "My order arrived damaged and I need a replacement."
}
```

Response includes `category`, `priority`, `confidence`, `requires_human_review`, and `justification`.

Requires `OPENAI_API_KEY` in `.env` (see `env.example`).

### Classify (ML)

```http
POST /tickets/classify/ml
```

Uses legacy body shape: `customer_id`, `message`, optional `order_id`.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp env.example .env   # set OPENAI_API_KEY for LLM routes

python -m app.train_classifier   # produces app/artifacts/category_classifier.joblib
pytest
uvicorn app.main:app --reload
```

OpenAPI UI: http://127.0.0.1:8000/docs

## Evaluation

```bash
python -m app.compare_classifiers
python -m app.compare_three_classifiers
python -m app.compare_three_classifiers --live-llm   # optional; uses OpenAI
```

Model metrics and methodology: [`docs/evaluation.md`](docs/evaluation.md).

## Knowledge base (Week 4)

Policy RAG with pgvector: ingest markdown policies, semantic search, and grounded answers.

```bash
docker compose up -d
export DATABASE_URL=postgresql+psycopg://opspilot:opspilot@localhost:5432/opspilot
python -m app.ingest_policies --include-superseded
python -m app.run_rag_evaluation
```

Endpoints: `POST /support/search`, `POST /support/answer`. Details: [`docs/knowledge-base.md`](docs/knowledge-base.md).

## Tech stack

Python 3.13, FastAPI, Pydantic v2, scikit-learn, OpenAI Python SDK, pgvector, pytest.

## License

Portfolio / demonstration project. Adapt and reuse with attribution.
