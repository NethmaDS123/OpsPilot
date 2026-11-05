from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from app.ml_training import training_data

messages = [
    item[0]
    for item in training_data
]
labels = [
    item[1]
    for item in training_data
]

X_train, X_test, y_train, y_test = train_test_split(
    messages,
    labels,
    test_size=5,
    random_state=42,
    stratify=labels,
)

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(),
    ),
    (
        "classifier",
        LogisticRegression(),
    ),
])

model.fit(
    X_train,
    y_train,
)

predictions = model.predict(
    X_test,
)

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0,
    )
)
