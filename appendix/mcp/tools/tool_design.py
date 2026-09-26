"""
Tool design helpers — validate and score tool definitions
for routability, schema quality, and error handling.
"""
import re
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ToolScore:
    name: str
    total: float
    details: dict[str, float]
    issues: list[str]
    suggestions: list[str]


class ToolValidator:
    """
    Audits tool definitions for common design problems.
    Scores description quality, schema completeness, and naming.
    """

    ROUTING_KEYWORDS = ["use when", "do not use", "returns", "do not", "instead"]

    def score(self, tool_fn: Callable) -> ToolScore:
        name   = tool_fn.__name__
        doc    = (tool_fn.__doc__ or "").strip()
        hints  = tool_fn.__annotations__

        scores   = {}
        issues   = []
        suggests = []

        # ── Description quality ───────────────────────────────────────────
        desc_score = 0.0

        if len(doc) < 30:
            issues.append("Description too short (<30 chars) — agents can't route on it.")
            suggests.append("Add: when to use, what it returns, what it doesn't handle.")
        else:
            desc_score += 0.3

        doc_lower = doc.lower()
        routing_found = [kw for kw in self.ROUTING_KEYWORDS if kw in doc_lower]
        if routing_found:
            desc_score += 0.4
        else:
            issues.append("No routing signal found (no 'use when', 'returns', 'do not use').")
            suggests.append("Add 'Use when...' and 'Do NOT use...' to the description.")

        if len(doc) > 500:
            issues.append("Description may be too long (>500 chars). Long descriptions dilute the routing signal.")

        desc_score = min(1.0, desc_score + (0.3 if routing_found else 0.0))
        scores["description"] = round(desc_score, 2)

        # ── Naming quality ────────────────────────────────────────────────
        if "_" not in name:
            issues.append(f"Tool name '{name}' has no namespace prefix. Use 'namespace_action_noun' pattern.")
            suggests.append(f"Rename to e.g. 'knowledge_{name}' or 'db_{name}'.")
            scores["naming"] = 0.4
        elif len(name) < 5:
            issues.append(f"Tool name '{name}' is too short — too generic.")
            scores["naming"] = 0.5
        else:
            scores["naming"] = 1.0

        # ── Schema quality ────────────────────────────────────────────────
        params = {k: v for k, v in hints.items() if k != "return"}
        if not params:
            scores["schema"] = 1.0  # no params — fine
        else:
            schema_score = 1.0
            for param, annotation in params.items():
                ann_str = str(annotation)
                if "str" in ann_str and param not in ("query", "text", "content", "description"):
                    # Free-form str params are high-ambiguity — check if constrained
                    if "Literal" not in ann_str and "Enum" not in ann_str:
                        if doc and param not in doc:
                            issues.append(
                                f"Parameter '{param}' is unconstrained str with no description. "
                                "Consider Literal[] or add format constraints to the docstring."
                            )
                            schema_score -= 0.1
            scores["schema"] = max(0.0, round(schema_score, 2))

        # ── Return type ───────────────────────────────────────────────────
        return_ann = hints.get("return", None)
        if return_ann is None:
            issues.append("No return type annotation. Add -> str, -> dict, or -> list.")
            scores["return_type"] = 0.5
        else:
            scores["return_type"] = 1.0

        # ── Total ─────────────────────────────────────────────────────────
        weights = {"description": 0.45, "naming": 0.20, "schema": 0.25, "return_type": 0.10}
        total = sum(scores.get(k, 0) * w for k, w in weights.items())

        return ToolScore(
            name=name,
            total=round(total, 2),
            details=scores,
            issues=issues,
            suggestions=suggests,
        )

    def report(self, tool_fn: Callable) -> str:
        s = self.score(tool_fn)
        grade = "A" if s.total >= 0.85 else "B" if s.total >= 0.70 else "C" if s.total >= 0.55 else "D"
        lines = [
            f"Tool: {s.name}",
            f"Score: {s.total:.0%} ({grade})",
            f"  description: {s.details.get('description', 0):.0%}",
            f"  naming:      {s.details.get('naming', 0):.0%}",
            f"  schema:      {s.details.get('schema', 0):.0%}",
            f"  return_type: {s.details.get('return_type', 0):.0%}",
        ]
        if s.issues:
            lines.append("\nIssues:")
            for issue in s.issues:
                lines.append(f"  ⚠  {issue}")
        if s.suggestions:
            lines.append("\nSuggestions:")
            for sug in s.suggestions:
                lines.append(f"  →  {sug}")
        return "\n".join(lines)


def describe_tool(
    when_to_use: str,
    returns: str,
    not_for: str = "",
    param_notes: dict[str, str] = None,
) -> str:
    """
    Helper to build a well-structured tool description.

    Args:
        when_to_use: Trigger condition ("Use when the user asks about X")
        returns:     Output format ("Returns list of matching items with ID and title")
        not_for:     Disambiguation ("Do NOT use for Y — use other_tool instead")
        param_notes: Per-parameter format hints {"param_name": "ISO 8601: '2026-06-23T14:30:00'"}

    Returns:
        Formatted description string ready to use as tool docstring.
    """
    parts = [when_to_use.strip()]
    if returns:
        parts.append(f"Returns: {returns.strip()}")
    if not_for:
        parts.append(f"Do NOT use for: {not_for.strip()}")
    if param_notes:
        for param, note in param_notes.items():
            parts.append(f"  {param}: {note}")
    return "\n\n".join(parts)
