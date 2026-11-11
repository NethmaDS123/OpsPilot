from collections import Counter

from app.labeled_dataset import LABELED_TICKETS

EXPECTED_CATEGORIES = {
    "billing",
    "refund",
    "technical",
    "delivery",
    "account",
    "other",
}


def test_dataset_size_and_categories():
    assert len(LABELED_TICKETS) >= 300
    counts = Counter(label for _, label in LABELED_TICKETS)
    assert set(counts) == EXPECTED_CATEGORIES
    for category in EXPECTED_CATEGORIES:
        assert counts[category] >= 40
