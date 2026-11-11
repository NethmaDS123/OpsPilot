"""Benchmark rule-based vs ML classifiers on a shared hold-out split."""

from __future__ import annotations

import statistics
import time

import joblib
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from app.labeled_dataset import LABELED_TICKETS
from app.models import TicketRequest
from app.rule_classifier import classify_ticket_rules
from app.train_classifier import MODEL_PATH, build_model

_ML_MODEL = None  # loaded once; compare script is CLI-only


def _get_ml_model():
    global _ML_MODEL
    if _ML_MODEL is None:
        if MODEL_PATH.is_file():
            _ML_MODEL = joblib.load(MODEL_PATH)
        else:
            model = build_model()
            messages = [item[0] for item in LABELED_TICKETS]
            labels = [item[1] for item in LABELED_TICKETS]
            model.fit(messages, labels)
            _ML_MODEL = model
    return _ML_MODEL


def _rule_category(message: str) -> str:
    result = classify_ticket_rules(
        TicketRequest(customer_id="eval", message=message),
    )
    return result.category


def _ml_category(message: str) -> str:
    return _get_ml_model().predict([message])[0]


def _latency_ms(fn, message: str, repeats: int = 50) -> float:
    timings: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn(message)
        timings.append((time.perf_counter() - start) * 1000)
    return statistics.mean(timings)


def main() -> None:
    messages = [item[0] for item in LABELED_TICKETS]
    labels = [item[1] for item in LABELED_TICKETS]

    _, X_test, _, y_test = train_test_split(
        messages,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    rule_preds = [_rule_category(m) for m in X_test]
    ml_preds = [_ml_category(m) for m in X_test]

    rule_acc = accuracy_score(y_test, rule_preds)
    ml_acc = accuracy_score(y_test, ml_preds)
    rule_f1 = f1_score(y_test, rule_preds, average="macro", zero_division=0)
    ml_f1 = f1_score(y_test, ml_preds, average="macro", zero_division=0)

    sample = X_test[0]
    rule_latency = _latency_ms(_rule_category, sample)
    ml_latency = _latency_ms(_ml_category, sample)

    print("Classifier comparison (same 20% hold-out, category only)")
    print()
    print(f"{'Metric':<22} {'Rule-based':<16} {'ML':<16}")
    print("-" * 54)
    print(f"{'Accuracy':<22} {rule_acc:<16.4f} {ml_acc:<16.4f}")
    print(f"{'Macro F1':<22} {rule_f1:<16.4f} {ml_f1:<16.4f}")
    print(f"{'Inference latency (ms)':<22} {rule_latency:<16.3f} {ml_latency:<16.3f}")
    print(f"{'Cost per prediction':<22} {'$0.00':<16} {'$0.00':<16}")
    print()
    print(
        "Cost assumes local CPU inference with no external API; "
        "rule-based cost is engineering maintenance only."
    )


if __name__ == "__main__":
    main()
