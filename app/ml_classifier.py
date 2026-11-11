"""
ML inference: sklearn predicts category; rules set priority and requires_human.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from app.models import (
    TicketCategory,
    TicketClassification,
    TicketRequest,
)
from app.rule_classifier import (
    detect_priority,
    matching_categories,
    requires_human_review,
)

ARTIFACT_PATH = (
    Path(__file__).resolve().parent / "artifacts" / "category_classifier.joblib"
)

ML_CONFIDENCE_THRESHOLD = 0.35


@lru_cache(maxsize=1)
def _load_model() -> Pipeline:
    if not ARTIFACT_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {ARTIFACT_PATH}. "
            "Run: python -m app.train_classifier"
        )
    return joblib.load(ARTIFACT_PATH)


def predict_category(message: str) -> tuple[TicketCategory, float]:
    model = _load_model()
    probabilities = model.predict_proba([message])[0]
    best_index = int(probabilities.argmax())
    category = model.classes_[best_index]
    confidence = float(probabilities[best_index])
    # Low softmax probability -> treat as unknown rather than a wrong label.
    if confidence < ML_CONFIDENCE_THRESHOLD:
        return "other", confidence
    return category, confidence


def classify_ticket(
    ticket: TicketRequest,
) -> TicketClassification:
    message_lower = ticket.message.lower()
    category, ml_confidence = predict_category(ticket.message)
    matched = matching_categories(message_lower)

    # Blend ML label with keyword overlap so ambiguous tickets escalate.
    rule_confident = len(ticket.message.strip()) >= 8 and (
        len(matched) <= 1
        or matched <= {"billing", "refund"}
        or category in matched
    )
    confident = ml_confidence >= ML_CONFIDENCE_THRESHOLD and rule_confident

    priority = detect_priority(message_lower, category)
    requires_human = requires_human_review(
        message_lower,
        matched,
        category,
        confident,
    )

    return TicketClassification(
        category=category,
        priority=priority,
        summary=ticket.message[:100],
        requires_human=requires_human,
    )
