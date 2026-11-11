# Architecture

## Classification strategies

OpsPilot implements three classifiers behind separate code paths so they can be benchmarked without rewriting the HTTP layer.

| Strategy | Module | Role |
|----------|--------|------|
| Rules | `app/rule_classifier.py` | Fast baseline; explicit keywords and precedence |
| ML | `app/ml_classifier.py` + `train_classifier.py` | Learned categories from labeled tickets |
| LLM | `app/services/llm_client.py` | General language understanding with structured output |

Production traffic uses the **LLM route** by default. The ML route remains for comparison and regression tests.

## LLM request path

1. **API** (`app/api/tickets.py`) validates `TicketClassifyRequest`.
2. **Classification service** normalizes input and calls the LLM client.
3. **OpenAI Responses API** returns output parsed into `LLMClassificationResult`.
4. **Safety layer** may force human review (injection patterns, sensitive topics, low confidence).
5. **Response** mapped to `TicketClassifyResponse`.

Configuration lives in `app/core/config.py` (API key, model, timeouts, confidence threshold).

## Design choices

- **Prompts in a dedicated module** — easier to review and version than inline strings.
- **`TicketClassifierClient` protocol** — swap OpenAI for another provider or a test double.
- **Model proposes, code disposes** — the LLM never executes refunds or account changes; backend rules enforce escalation.
- **Structured outputs** — Pydantic schema at the provider boundary reduces brittle JSON parsing.

## Testing

Tests override `get_classifier_client` with `FakeClassifierClient` (`tests/conftest.py`) so CI does not require network access or API keys.
