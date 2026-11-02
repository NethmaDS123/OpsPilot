# OpsPilot

Fictional customer-support operations platform. This repository currently implements an extended, rule-based ticket classifier exposed through a FastAPI service.

## Architecture

The application is split into three layers:

| Module | Responsibility |
|--------|----------------|
| `app/models.py` | Request/response schemas and allowed enum values |
| `app/classifier.py` | Pure classification logic (no HTTP or I/O) |
| `app/main.py` | HTTP routes that validate input and call the classifier |

```
POST /tickets/classify
        │
        ▼
   TicketRequest  (Pydantic validation)
        │
        ▼
   classify_ticket()
        │
        ▼
   TicketResponse (ticket_id + TicketClassification)
```

Keeping classification as a plain function makes it easy to unit test, swap in an LLM later, and run offline evaluations without starting a server.

## Data models

Models use **Pydantic `BaseModel`** with **`Literal` types** for `category` and `priority`. That encodes the allowed values in the type system so invalid states (for example `category="refundmaybe"`) are rejected at validation time rather than deep in business logic.

- **`TicketRequest`**: inbound ticket (`customer_id`, `message`, optional `order_id`).
- **`TicketClassification`**: structured outcome (`category`, `priority`, `summary`, `requires_human`).
- **`TicketResponse`**: API envelope with a generated `ticket_id` and nested classification.

Field constraints on `message` (1–5000 characters) and `summary` (1–300 characters) enforce basic input bounds at the boundary.

## Validation

Validation happens in two places:

1. **Inbound API body** — FastAPI parses JSON into `TicketRequest`. Missing fields, empty messages, or overlong text produce **HTTP 422** with Pydantic error details.
2. **Outbound response** — `response_model=TicketResponse` ensures the handler returns data that matches the declared schema.

The classifier assumes it receives an already-valid `TicketRequest`; it does not re-validate string lengths.

## Error handling

- **Client errors (422)**: Invalid or malformed ticket payloads are rejected by Pydantic before `classify_ticket` runs.
- **Server errors**: Unhandled exceptions would surface as 500 responses; the classifier path is deterministic and has no external dependencies, so failures here would indicate a programming bug rather than a transient service outage.

There is no custom exception hierarchy yet; that can be added when we integrate databases and external tools.

## Classifier behaviour

The classifier lowercases the message and applies **keyword and phrase rules** grouped by category. It then:

1. Collects all categories that match.
2. Resolves a single category using severity precedence (security and billing issues win over generic matches).
3. Assigns **priority** (`low`, `normal`, `high`, `urgent`) from explicit rules aligned with the assignment examples.
4. Sets **`requires_human=True`** when:
   - The account is reported compromised.
   - A serious financial issue is reported (duplicate charge, fraud, stolen card, etc.).
   - The customer asks for an action the automated system must not take (e.g. chargeback, legal escalation).
   - The outcome is **not confident** — multiple unrelated categories match, the message is too short/vague, or `other` is chosen without a clear general-inquiry signal.

**Confidence fallback (pre-LLM):** There is no model probability score. Confidence is approximated with heuristics: multi-category conflicts (except billing/refund overlap), very short messages, and weak lone substring matches (e.g. negated “refund”) trigger escalation via `requires_human=True`.

## Testing

Tests use **pytest**.

- **`tests/test_classifier.py`** — Unit tests for classification: normal cases, edge cases (negated refund, ambiguity), escalation, and direct Pydantic validation of invalid `TicketRequest` instances.
- **`tests/test_api.py`** — Integration-style tests with FastAPI’s `TestClient`: health check, successful classify, and 422 on empty message.

Run:

```bash
source .venv/bin/activate
pytest
```

## Current limitations

- **Keyword matching** misses intent, sarcasm, and negation in general; only a few edge cases (e.g. refund + tracking) are handled explicitly.
- **Categories in the type system** include values the rules rarely assign alone (e.g. billing vs refund overlap is simplified).
- **No persistence** — `ticket_id` is generated per request and not stored.
- **No LLM, tools, or human workflow** — escalation is a boolean flag only; there is no queue or approval UI yet.
- **English-only** phrases; no locale or stemming.

These limits are intentional for Assignment 1: establish a testable baseline before replacing or augmenting rules with an AI classifier in later lessons.

## Run the API

```bash
source .venv/bin/activate
uvicorn app.main:app --reload
```

Interactive docs: http://127.0.0.1:8000/docs
