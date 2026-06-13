"""Request validation and prompt injection detection."""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)


class LLMRequestValidator:
    """Validate incoming LLM requests."""

    INJECTION_PATTERNS = [
        r"ignore previous instructions",
        r"forget all.*rules",
        r"you are now.*without restrictions",
        r"output your.*system prompt",
        r"bypass.*safety",
    ]

    def validate(
        self,
        messages: list[dict],
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> ValidationResult:
        errors = []

        if not messages:
            errors.append("Messages cannot be empty")

        for i, msg in enumerate(messages):
            if "role" not in msg or "content" not in msg:
                errors.append(f"Message {i} missing role or content")
            elif msg["role"] not in ("system", "user", "assistant"):
                errors.append(f"Invalid role in message {i}: {msg['role']}")
            elif self._detect_injection(msg.get("content", "")):
                errors.append(f"Potential prompt injection detected in message {i}")

        if temperature < 0 or temperature > 2:
            errors.append(f"Temperature {temperature} out of range [0, 2]")
        if max_tokens < 1 or max_tokens > 128_000:
            errors.append(f"max_tokens {max_tokens} out of range [1, 128000]")

        return ValidationResult(valid=len(errors) == 0, errors=errors)

    def _detect_injection(self, text: str) -> bool:
        text_lower = text.lower()
        return any(re.search(p, text_lower) for p in self.INJECTION_PATTERNS)


def validate_request(messages: list[dict], **kwargs) -> ValidationResult:
    """Convenience function for request validation."""
    return LLMRequestValidator().validate(messages, **kwargs)
