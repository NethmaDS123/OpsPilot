from app.models import (
    TicketCategory,
    TicketPriority,
    TicketClassification,
    TicketRequest,
)

# Ordered from most specific / highest severity to broad keyword buckets.
_CATEGORY_KEYWORDS: dict[TicketCategory, tuple[str, ...]] = {
    "account": (
        "compromised",
        "been hacked",
        "hacked my account",
        "unauthorized access",
        "someone logged in",
        "update my account",
        "change my name",
        "account name",
        "how do i update",
        "profile settings",
        "reset my password",
    ),
    "billing": (
        "charged twice",
        "duplicate charge",
        "double charged",
        "charged two times",
        "wrong charge",
        "incorrect charge",
        "unauthorized charge",
        "fraudulent charge",
        "stolen card",
    ),
    "refund": (
        "want a refund",
        "need a refund",
        "request a refund",
        "refund my order",
        "money back",
        "return my money",
    ),
    "technical": (
        "broken",
        "damaged",
        "not working",
        "doesn't work",
        "does not work",
        "error message",
        "keeps crashing",
    ),
    "delivery": (
        "hasn't arrived",
        "has not arrived",
        "not arrived",
        "where is my package",
        "shipping delay",
        "tracking number",
        "tracking",
        "delivery status",
        "still waiting for my order",
    ),
}

_ACCOUNT_COMPROMISE = (
    "compromised",
    "been hacked",
    "hacked my account",
    "unauthorized access",
    "someone logged in",
)

_SERIOUS_FINANCIAL = (
    "charged twice",
    "duplicate charge",
    "double charged",
    "charged two times",
    "unauthorized charge",
    "fraudulent charge",
    "stolen card",
)

_UNAUTHORIZED_ACTIONS = (
    "chargeback",
    "file a dispute with my bank",
    "speak to a lawyer",
    "legal action",
    "sue your company",
)

_REFUND_KEYWORDS = _CATEGORY_KEYWORDS["refund"] + ("refund",)

_REFUND_NEGATIONS = (
    "don't want a refund",
    "do not want a refund",
    "don't need a refund",
    "not asking for a refund",
)

_LOW_PRIORITY_ACCOUNT = (
    "how do i update",
    "update my account",
    "change my name",
    "account name",
    "profile settings",
)

_GENERAL_INQUIRY = (
    "general question",
    "quick question",
    "just wondering",
    "hello",
    "hi there",
)


def _contains_any(message: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in message for phrase in phrases)


def _refund_requested(message: str) -> bool:
    if _contains_any(message, _REFUND_NEGATIONS):
        return False
    return _contains_any(message, _REFUND_KEYWORDS)


def _matching_categories(message: str) -> set[TicketCategory]:
    matched: set[TicketCategory] = set()
    if _refund_requested(message):
        matched.add("refund")
    for category, keywords in _CATEGORY_KEYWORDS.items():
        if category == "refund":
            continue
        if _contains_any(message, keywords):
            matched.add(category)
    return matched


def _resolve_category(
    message: str,
    matched: set[TicketCategory],
) -> TicketCategory:
    if _contains_any(message, _ACCOUNT_COMPROMISE):
        return "account"
    if _contains_any(message, _SERIOUS_FINANCIAL):
        return "billing"
    if "refund" in matched and "billing" not in matched:
        return "refund"
    if len(matched) == 1:
        return next(iter(matched))
    if len(matched) > 1:
        # Prefer the most actionable category when keywords overlap.
        precedence: tuple[TicketCategory, ...] = (
            "billing",
            "account",
            "refund",
            "delivery",
            "technical",
        )
        for category in precedence:
            if category in matched:
                return category
    if _contains_any(message, _GENERAL_INQUIRY):
        return "other"
    return "other"


def _detect_priority(
    message: str,
    category: TicketCategory,
) -> TicketPriority:
    if _contains_any(message, _ACCOUNT_COMPROMISE):
        return "urgent"
    if _contains_any(message, _SERIOUS_FINANCIAL):
        return "high"
    if category == "delivery" or category == "technical" or category == "refund":
        return "normal"
    if category == "account" and _contains_any(message, _LOW_PRIORITY_ACCOUNT):
        return "low"
    if category == "other" and _contains_any(message, _GENERAL_INQUIRY):
        return "low"
    if category == "billing":
        return "normal"
    return "normal"


def _is_confident(
    message: str,
    matched: set[TicketCategory],
    category: TicketCategory,
) -> bool:
    stripped = message.strip()
    if len(stripped) < 8:
        return False
    if len(matched) > 1 and not (
        matched <= {"billing", "refund"} or matched <= {"refund", "billing"}
    ):
        return False
    if category == "other" and not matched:
        if _contains_any(message, _GENERAL_INQUIRY):
            return True
        return False
    if category == "refund" and not _refund_requested(message):
        return False
    return True


def _requires_human(
    message: str,
    matched: set[TicketCategory],
    category: TicketCategory,
    confident: bool,
) -> bool:
    if _contains_any(message, _ACCOUNT_COMPROMISE):
        return True
    if _contains_any(message, _SERIOUS_FINANCIAL):
        return True
    if _contains_any(message, _UNAUTHORIZED_ACTIONS):
        return True
    if not confident:
        return True
    return False


def classify_ticket(
    ticket: TicketRequest,
) -> TicketClassification:
    message = ticket.message.lower()
    matched = _matching_categories(message)
    category = _resolve_category(message, matched)
    priority = _detect_priority(message, category)
    confident = _is_confident(message, matched, category)
    requires_human = _requires_human(message, matched, category, confident)

    return TicketClassification(
        category=category,
        priority=priority,
        summary=ticket.message[:100],
        requires_human=requires_human,
    )
