"""Output guardrails — format validation, hallucination detection, toxicity check."""

import json
import re
from dataclasses import dataclass, field


@dataclass
class OutputGuardrailResult:
    allowed: bool
    reason: str = ""
    issues: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class OutputGuardrail:
    """Check LLM output for quality and safety issues."""

    UNCERTAINTY_PHRASES = [
        "i think", "probably", "might be",
        "not sure", "could be", "perhaps",
    ]

    TOXIC_WORDS = ["stupid", "idiot", "hate", "kill", "die"]

    def check(self, text: str, expected_format: str = None) -> OutputGuardrailResult:
        issues = []
        metadata = {
            "length": len(text),
            "contains_code": "```" in text,
            "contains_urls": "http" in text,
        }

        if expected_format == "json":
            valid, err = self._validate_json(text)
            if not valid:
                issues.append(err)

        hall_score = self._hallucination_score(text)
        if hall_score > 0.7:
            issues.append(f"High hallucination risk: {hall_score:.2f}")
            metadata["hallucination_score"] = hall_score

        tox_score = self._toxicity_score(text)
        if tox_score > 0.5:
            issues.append(f"Toxic content detected: {tox_score:.2f}")
            metadata["toxicity_score"] = tox_score

        return OutputGuardrailResult(
            allowed=len(issues) == 0,
            reason=issues[0] if issues else "",
            issues=issues,
            metadata=metadata,
        )

    def _validate_json(self, text: str) -> tuple[bool, str]:
        try:
            if "```json" in text:
                start = text.find("```json") + 7
                end = text.find("```", start)
                text = text[start:end].strip()
            elif "```" in text:
                start = text.find("```") + 3
                end = text.find("```", start)
                text = text[start:end].strip()
            json.loads(text)
            return True, ""
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {e}"

    def _hallucination_score(self, text: str) -> float:
        text_lower = text.lower()
        count = sum(1 for phrase in self.UNCERTAINTY_PHRASES if phrase in text_lower)
        return min(count / 5.0, 1.0)

    def _toxicity_score(self, text: str) -> float:
        text_lower = text.lower()
        count = sum(1 for word in self.TOXIC_WORDS if word in text_lower)
        return min(count / 10.0, 1.0)


def check_output(text: str, **kwargs) -> OutputGuardrailResult:
    """Convenience function for output checking."""
    return OutputGuardrail().check(text, **kwargs)
