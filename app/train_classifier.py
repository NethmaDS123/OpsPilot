"""Train and evaluate the TF-IDF + logistic regression category model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.labeled_dataset import LABELED_TICKETS

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "category_classifier.joblib"
METRICS_PATH = ARTIFACT_DIR / "training_metrics.json"


def build_model() -> Pipeline:
    # Pipeline ensures the same TF-IDF transform at train and predict time.
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
            (
                "classifier",
                LogisticRegression(max_iter=1000, random_state=42),
            ),
        ]
    )


def main() -> None:
    messages = [item[0] for item in LABELED_TICKETS]
    labels = [item[1] for item in LABELED_TICKETS]

    X_train, X_test, y_train, y_test = train_test_split(
        messages,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    model = build_model()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision_macro": precision_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "f1_macro": f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "test_size": len(y_test),
        "train_size": len(y_train),
    }

    print("Hold-out metrics")
    print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    print(f"  Precision:  {metrics['precision_macro']:.4f} (macro)")
    print(f"  Recall:     {metrics['recall_macro']:.4f} (macro)")
    print(f"  Macro F1:   {metrics['f1_macro']:.4f}")
    print()
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix (rows=true, cols=predicted)")
    labels_order = sorted(set(labels))
    matrix = confusion_matrix(y_test, predictions, labels=labels_order)
    print(f"Labels: {labels_order}")
    print(matrix)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nSaved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
