"""30-question retrieval benchmark with expected policy document IDs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RAGEvalCase:
    question: str
    company_id: str
    expected_document_id: str
    expect_abstain: bool = False


RAG_EVAL_CASES: list[RAGEvalCase] = [
    RAGEvalCase("How long do I have to request a full refund on standard products?", "acme-corp", "acme-refunds-v2"),
    RAGEvalCase("Are final sale items refundable?", "acme-corp", "acme-refunds-v2"),
    RAGEvalCase("When do refunds above $500 need supervisor approval?", "acme-corp", "acme-refunds-v2"),
    RAGEvalCase("What is the return window for physical goods?", "acme-corp", "acme-returns-v1"),
    RAGEvalCase("Is there a restocking fee for opened electronics?", "acme-corp", "acme-returns-v1"),
    RAGEvalCase("Who pays return shipping for change-of-mind returns?", "acme-corp", "acme-returns-v1"),
    RAGEvalCase("How soon must I report visible shipping damage?", "acme-corp", "acme-damaged-goods-v1"),
    RAGEvalCase("What photos are required for a damage claim?", "acme-corp", "acme-damaged-goods-v1"),
    RAGEvalCase("Can I get a repair instead of replacement for expensive items?", "acme-corp", "acme-damaged-goods-v1"),
    RAGEvalCase("When can I request a courtesy credit for late delivery?", "acme-corp", "acme-delivery-delays-v1"),
    RAGEvalCase("How long can tracking be stagnant before opening a delay ticket?", "acme-corp", "acme-delivery-delays-v1"),
    RAGEvalCase("What happens after 10 business days past estimated delivery?", "acme-corp", "acme-delivery-delays-v1"),
    RAGEvalCase("What counts as a missing package?", "acme-corp", "acme-missing-orders-v1"),
    RAGEvalCase("Do I need a police report for porch theft over $150?", "acme-corp", "acme-missing-orders-v1"),
    RAGEvalCase("How many courtesy replacements are allowed per year?", "acme-corp", "acme-missing-orders-v1"),
    RAGEvalCase("How long are password reset links valid?", "acme-corp", "acme-account-access-v2"),
    RAGEvalCase("What ID is needed if I lost email access?", "acme-corp", "acme-account-access-v2"),
    RAGEvalCase("Can support disable MFA without proof?", "acme-corp", "acme-account-access-v2"),
    RAGEvalCase("How long are declined-card orders held?", "acme-corp", "acme-payment-failures-v1"),
    RAGEvalCase("What is the subscription grace period after payment failure?", "acme-corp", "acme-payment-failures-v1"),
    RAGEvalCase("What happens if I file a chargeback?", "acme-corp", "acme-payment-failures-v1"),
    RAGEvalCase("Can I cancel an order that already shipped?", "acme-corp", "acme-cancellations-v1"),
    RAGEvalCase("Is there a refund for partial months on monthly subscriptions?", "acme-corp", "acme-cancellations-v1"),
    RAGEvalCase("What is the annual plan cancellation refund window?", "acme-corp", "acme-cancellations-v1"),
    RAGEvalCase("What is the warranty on electronics?", "acme-corp", "acme-warranty-v1"),
    RAGEvalCase("What is the deductible for Acme Care accidental damage?", "acme-corp", "acme-warranty-v1"),
    RAGEvalCase("When does the lemon policy apply?", "acme-corp", "acme-warranty-v1"),
    RAGEvalCase("How fast do I get a complaint case ID?", "acme-corp", "acme-complaints-v1"),
    RAGEvalCase("What is the escalation path for unresolved complaints?", "acme-corp", "acme-complaints-v1"),
    RAGEvalCase("Do Globex refunds go to the original card?", "globex-inc", "globex-refunds-v1"),
    RAGEvalCase(
        "What is Acme's policy on Mars colony shipping?",
        "acme-corp",
        "",
        expect_abstain=True,
    ),
]
