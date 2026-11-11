# Model evaluation

## Dataset

`app/labeled_dataset.py` provides **300+ synthetic, realistic** labeled tickets across six categories: billing, refund, technical, delivery, account, and other. Messages use varied templates (order IDs, tracking numbers, dates) rather than single-token edits.

## Traditional ML

**Pipeline:** `TfidfVectorizer` (1–2 grams, `min_df=2`) + `LogisticRegression`.

**Split:** 80% train / 20% test, stratified, `random_state=42`.

**Artifacts:** `app/artifacts/category_classifier.joblib`, `training_metrics.json`.

Hold-out performance (representative run): ~**79% accuracy**, ~**78% macro F1** on this synthetic corpus.

The ML model predicts **category only**. Priority and `requires_human` use shared rule heuristics in `rule_classifier.py`.

## Comparing classifiers

| Script | Compares |
|--------|----------|
| `app/compare_classifiers.py` | Rules vs ML |
| `app/compare_three_classifiers.py` | Rules vs ML vs LLM |

All scripts use the **same stratified hold-out split** for fair comparison. Metrics include accuracy, macro F1, recall on billing/account/refund, latency, failure rate, and human-review rate (where applicable).

On this dataset, **ML typically outperforms rules** on category accuracy because many paraphrases never hit keyword lists. **Rules remain faster** and more predictable on edge cases. **LLM** trade-offs depend on provider, prompt, and whether you run live API calls (`--live-llm`) or the deterministic fake client used in tests.

## Next steps

- Expand labeled data with production-like distribution and human review.
- Calibrate confidence and track token cost per ticket in production logs.
- Add retries/backoff and a deliberate degraded mode (e.g. queue for human or fall back to rules).
