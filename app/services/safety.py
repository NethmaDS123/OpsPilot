"""Input hygiene and heuristic checks before/after LLM classification."""

import re

# Substrings that suggest the user is trying to manipulate the model or exfiltrate secrets.

_INJECTION_PATTERNS = (
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"disregard\s+(the\s+)?(system|above)\s+(prompt|instructions)",
    r"you\s+are\s+now\s+",
    r"reveal\s+(the\s+|your\s+)?(system\s+)?prompt",
    r"print\s+(your\s+)?(system\s+)?instructions",
    r"api\s*key",
    r"password\s+for\s+all\s+users",
    r"<\s*script",
)

# Topics that always warrant human review regardless of model confidence.
_SENSITIVE_PATTERNS = (
    r"\bhacked\b",
    r"\bcompromised\b",
    r"unauthorized\s+access",
    r"\bfraud\b",
    r"stolen\s+card",
    r"chargeback",
    r"legal\s+action",
    r"\bsue\b",
)

_injection_re = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)
_sensitive_re = re.compile("|".join(_SENSITIVE_PATTERNS), re.IGNORECASE)


def normalize_message(message: str) -> str:
    cleaned = message.replace("\x00", "").strip()
    if not cleaned:
        raise ValueError("Message must not be empty or whitespace only.")
    return cleaned


def detect_prompt_injection(message: str) -> bool:
    return bool(_injection_re.search(message))


def detect_sensitive_content(message: str) -> bool:
    return bool(_sensitive_re.search(message))
