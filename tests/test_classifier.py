import pytest
from pydantic import ValidationError

from app.rule_classifier import classify_ticket_rules
from app.models import TicketRequest


def test_account_name_update_low_priority():
    ticket = TicketRequest(
        customer_id="cust_1",
        message="How do I update my account name?",
    )
    result = classify_ticket_rules(ticket)
    assert result.category == "account"
    assert result.priority == "low"
    assert result.requires_human is False


def test_delivery_not_arrived_normal_priority():
    ticket = TicketRequest(
        customer_id="cust_1",
        message="My package hasn't arrived yet.",
    )
    result = classify_ticket_rules(ticket)
    assert result.category == "delivery"
    assert result.priority == "normal"
    assert result.requires_human is False


def test_duplicate_charge_high_billing_escalates():
    ticket = TicketRequest(
        customer_id="cust_1",
        message="I was charged twice for the same order.",
        order_id="ord_99",
    )
    result = classify_ticket_rules(ticket)
    assert result.category == "billing"
    assert result.priority == "high"
    assert result.requires_human is True


def test_compromised_account_urgent_escalates():
    ticket = TicketRequest(
        customer_id="cust_1",
        message="My account has been compromised and someone logged in.",
    )
    result = classify_ticket_rules(ticket)
    assert result.category == "account"
    assert result.priority == "urgent"
    assert result.requires_human is True


def test_refund_request_category():
    ticket = TicketRequest(
        customer_id="cust_1",
        message="I want a refund for order ord_12 please.",
        order_id="ord_12",
    )
    result = classify_ticket_rules(ticket)
    assert result.category == "refund"
    assert result.priority == "normal"
    assert result.requires_human is False


def test_technical_issue():
    ticket = TicketRequest(
        customer_id="cust_1",
        message="The device arrived damaged and is not working.",
    )
    result = classify_ticket_rules(ticket)
    assert result.category == "technical"
    assert result.priority == "normal"
    assert result.requires_human is False


def test_ambiguous_multi_category_escalates():
    ticket = TicketRequest(
        customer_id="cust_1",
        message=(
            "My package hasn't arrived and the app keeps crashing "
            "with an error message."
        ),
    )
    result = classify_ticket_rules(ticket)
    assert result.requires_human is True


def test_vague_short_message_escalates():
    ticket = TicketRequest(
        customer_id="cust_1",
        message="help!!!",
    )
    result = classify_ticket_rules(ticket)
    assert result.category == "other"
    assert result.requires_human is True


def test_unauthorized_action_escalates():
    ticket = TicketRequest(
        customer_id="cust_1",
        message="I am going to file a chargeback with my bank.",
    )
    result = classify_ticket_rules(ticket)
    assert result.requires_human is True


def test_general_inquiry_other_low_confidence_ok():
    ticket = TicketRequest(
        customer_id="cust_1",
        message="Hello, I have a general question about your store hours.",
    )
    result = classify_ticket_rules(ticket)
    assert result.category == "other"
    assert result.priority == "low"
    assert result.requires_human is False


def test_negated_refund_substring_escalates():
    ticket = TicketRequest(
        customer_id="cust_1",
        message="I don't want a refund, I only need tracking.",
    )
    result = classify_ticket_rules(ticket)
    assert result.category == "delivery"
    assert result.requires_human is False


def test_ticket_request_rejects_empty_message():
    with pytest.raises(ValidationError):
        TicketRequest(customer_id="cust_1", message="")


def test_ticket_request_rejects_overlong_message():
    with pytest.raises(ValidationError):
        TicketRequest(customer_id="cust_1", message="x" * 5001)
