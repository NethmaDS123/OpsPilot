from pathlib import Path

import pytest

from app.classifier import classify_ticket
from app.ml_classifier import ARTIFACT_PATH
from app.models import TicketRequest

pytestmark = pytest.mark.skipif(
    not ARTIFACT_PATH.is_file(),
    reason="Run python -m app.train_classifier to create the model artifact",
)


def test_ml_billing_duplicate_charge():
    ticket = TicketRequest(
        customer_id="cust_1",
        message="My payment was taken two times on my statement",
    )
    result = classify_ticket(ticket)
    assert result.category == "billing"


def test_ml_delivery_tracking():
    ticket = TicketRequest(
        customer_id="cust_1",
        message="Where is my tracking number for ord_5510?",
    )
    result = classify_ticket(ticket)
    assert result.category == "delivery"


def test_ml_still_escalates_serious_billing():
    ticket = TicketRequest(
        customer_id="cust_1",
        message="I was charged twice for my order",
        order_id="ord_456",
    )
    result = classify_ticket(ticket)
    assert result.category == "billing"
    assert result.priority == "high"
    assert result.requires_human is True
