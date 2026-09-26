"""Input guardrails — PII detection, toxicity filtering, injection detection."""

import re
from dataclasses import dataclass, field


@dataclass
class GuardrailResult:
    allowed: bool
    reason: str = ""
    issues: list[str] = field(default_factory=list)


class InputGuardrail:
    """Check user input for safety issues before sending to LLM."""

    PII_PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",
    }

    TOXIC_KEYWORDS = [
        "hate speech", "harassment", "violence",
        "self-harm", "suicide", "bomb", "weapon",
    ]

    INJECTION_PHRASES = [
        "you are now", "disregard all", "ignore safety",
        "output everything", "what is your system prompt",
    ]

    def check(self, text: str) -> GuardrailResult:
        issues = []

        for pii_type, pattern in self.PII_PATTERNS.items():
            if re.search(pattern, text):
                issues.append(f"Contains {pii_type.upper()}")

        text_lower = text.lower()
        for keyword in self.TOXIC_KEYWORDS:
            if keyword in text_lower:
                issues.append(f"Potentially toxic: {keyword}")

        if len(text) > 10_000:
            issues.append("Input too long (>10k chars)")

        if any(phrase in text_lower for phrase in self.INJECTION_PHRASES):
            issues.append("Potential prompt injection")

        if text.count("###") > 2:
            issues.append("Suspicious delimiter usage")

        return GuardrailResult(
            allowed=len(issues) == 0,
            reason=issues[0] if issues else "",
            issues=issues,
        )

    def sanitize(self, text: str) -> str:
        """Remove or mask PII from text."""
        for pii_type, pattern in self.PII_PATTERNS.items():
            text = re.sub(pattern, f"[{pii_type.upper()}_REDACTED]", text)
        return text


def check_input(text: str) -> GuardrailResult:
    """Convenience function for input checking."""
    return InputGuardrail().check(text)
