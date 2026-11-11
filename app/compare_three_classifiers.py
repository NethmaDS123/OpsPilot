"""Benchmark rule-based, ML, and LLM classifiers on a shared hold-out split."""

from __future__ import annotations

import argparse
import statistics
import time

import joblib
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split

from app.labeled_dataset import LABELED_TICKETS
from app.models import TicketRequest
from app.rule_classifier import classify_ticket_rules
from app.schemas.classification import TicketClassifyRequest
from app.services.classification import classify_ticket_with_llm_safe
from app.train_classifier import MODEL_PATH, build_model
from tests.conftest import FakeClassifierClient

IMPORTANT_CATEGORIES = ("billing", "account", "refund")

_ML_MODEL = None


def _get_ml_model():
    global _ML_MODEL
    if _ML_MODEL is None:
        _ML_MODEL = joblib.load(MODEL_PATH)
    return _ML_MODEL


def _rule_label(message: str) -> str:
    return classify_ticket_rules(
        TicketRequest(customer_id="eval", message=message),
    ).category


def _ml_label(message: str) -> str:
    return _get_ml_model().predict([message])[0]


def _llm_label(message: str, live: bool) -> tuple[str, bool]:
    # Default fake client keeps benchmarks free; --live-llm calls OpenAI (paid).
    if live:
        from app.services.llm_client import build_llm_client

        client = build_llm_client()
    else:
        client = FakeClassifierClient()
    response = classify_ticket_with_llm_safe(
        TicketClassifyRequest(ticket_id="eval", message=message),
        client,
    )
    return response.category, response.requires_human_review


def _latency_ms(fn, message: str, repeats: int = 25) -> float:
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn(message)
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.mean(samples)


def _evaluate(
    name: str,
    y_true: list[str],
    y_pred: list[str],
    latency_sample: str,
    predict_fn,
    human_flags: list[bool] | None = None,
    failures: int = 0,
    token_cost_per_100: float = 0.0,
) -> None:
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    recalls = {
        cat: recall_score(
            [1 if y == cat else 0 for y in y_true],
            [1 if p == cat else 0 for p in y_pred],
            zero_division=0,
        )
        for cat in IMPORTANT_CATEGORIES
    }
    latency = _latency_ms(predict_fn, latency_sample)
    review_rate = (
        sum(1 for flag in human_flags if flag) / len(human_flags)
        if human_flags
        else 0.0
    )
    failure_rate = failures / len(y_true) if y_true else 0.0

    print(f"\n=== {name} ===")
    print(f"Accuracy:              {acc:.4f}")
    print(f"Macro F1:              {macro_f1:.4f}")
    for cat in IMPORTANT_CATEGORIES:
        print(f"Recall ({cat}):      {recalls[cat]:.4f}")
    print(f"Latency (ms):          {latency:.3f}")
    print(f"Cost per 100 tickets: ${token_cost_per_100:.2f}")
    print(f"Failure rate:          {failure_rate:.4f}")
    print(f"Human review rate:     {review_rate:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live-llm",
        action="store_true",
        help="Call OpenAI instead of the deterministic fake client (incurs cost).",
    )
    args = parser.parse_args()

    messages = [item[0] for item in LABELED_TICKETS]
    labels = [item[1] for item in LABELED_TICKETS]
    _, X_test, _, y_test = train_test_split(
        messages,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    rule_preds = [_rule_label(m) for m in X_test]
    ml_preds = [_ml_label(m) for m in X_test]

    llm_preds: list[str] = []
    human_flags: list[bool] = []
    llm_failures = 0
    for message in X_test:
        try:
            category, review = _llm_label(message, live=args.live_llm)
            llm_preds.append(category)
            human_flags.append(review)
        except Exception:
            llm_failures += 1
            llm_preds.append("other")
            human_flags.append(True)

    sample = X_test[0]
    _evaluate(
        "Rule-based",
        y_test,
        rule_preds,
        sample,
        _rule_label,
        human_flags=[classify_ticket_rules(TicketRequest(customer_id="e", message=m)).requires_human for m in X_test],
    )
    _evaluate("Traditional ML", y_test, ml_preds, sample, _ml_label)
    _evaluate(
        "LLM" + (" (live)" if args.live_llm else " (fake client)"),
        y_test,
        llm_preds,
        sample,
        lambda m: _llm_label(m, live=args.live_llm)[0],
        human_flags=human_flags,
        failures=llm_failures,
        token_cost_per_100=0.15 if args.live_llm else 0.0,
    )


if __name__ == "__main__":
    main()
