"""Input/output guardrails — Module 10 applied."""
import re
from dataclasses import dataclass

INJECTION_PATTERNS = [
    r"ignore (all |previous |prior )?(instructions?|rules?|guidelines?)",
    r"(forget|disregard|bypass).{0,20}(instructions?|system)",
    r"you are now",
    r"act as (if )?you (have no|are without)",
    r"(reveal|show|output|print).{0,20}system prompt",
]

UNSAFE_OUTPUT_PATTERNS = [
    r"\b(ssn|social security number)\b",
    r"\b\d{3}-\d{2}-\d{4}\b",          # SSN format
    r"\b4[0-9]{12}(?:[0-9]{3})?\b",    # Visa card number pattern
]


@dataclass
class GuardResult:
    allowed: bool
    reason: str = ""


def check_input(text: str) -> GuardResult:
    text_lower = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text_lower):
            return GuardResult(allowed=False, reason=f"Potential prompt injection: '{pattern}'")
    if len(text) > 4000:
        return GuardResult(allowed=False, reason="Input exceeds 4000 character limit")
    return GuardResult(allowed=True)


def check_output(text: str) -> GuardResult:
    for pattern in UNSAFE_OUTPUT_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return GuardResult(allowed=False, reason="Output contains potentially sensitive data")
    return GuardResult(allowed=True)
