"""The normalized shapes the provider layer speaks in.

Nothing here mentions a vendor. A Message, a ToolSpec and a Response mean the
same thing whichever model answers, which is what lets the Flagship Agent be
pointed at a different provider without being rewritten.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

# ── Roles and stop reasons ────────────────────────────────────────────────────

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_TOOL = "tool"

STOP_END_TURN = "end_turn"
STOP_TOOL_USE = "tool_use"
STOP_MAX_TOKENS = "max_tokens"
STOP_STOP_SEQUENCE = "stop_sequence"
STOP_REFUSAL = "refusal"
STOP_OTHER = "other"


# ── Tools ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ToolSpec:
    """A tool as the agent declares it: a name, a description, a JSON schema."""

    name: str
    description: str
    input_schema: Mapping[str, Any]


@dataclass(frozen=True)
class ToolCall:
    """A tool call the model asked for, in normalized form."""

    id: str
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolResult:
    """The outcome of running a ToolCall, ready to send back.

    ``name`` is carried alongside ``call_id`` because Gemini identifies a tool
    response by name while the others identify it by call id. Keeping both in
    the normalized form is what lets a transcript round-trip to any provider.
    """

    call_id: str
    name: str
    content: str
    is_error: bool = False


# ── Messages ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Message:
    """One turn. A turn is text, tool calls, tool results, or text plus calls."""

    role: str
    text: Optional[str] = None
    tool_calls: Tuple[ToolCall, ...] = ()
    tool_results: Tuple[ToolResult, ...] = ()

    @classmethod
    def user(cls, text: str) -> "Message":
        return cls(role=ROLE_USER, text=text)

    @classmethod
    def assistant(
        cls, text: Optional[str] = None, tool_calls: Sequence[ToolCall] = ()
    ) -> "Message":
        return cls(role=ROLE_ASSISTANT, text=text, tool_calls=tuple(tool_calls))

    @classmethod
    def tool(cls, results: Sequence[ToolResult]) -> "Message":
        return cls(role=ROLE_TOOL, tool_results=tuple(results))


# ── Options ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CallOptions:
    """Intent-level options. What the caller wants, not what a vendor accepts.

    ``effort`` is a level from ``provider.registry.EFFORT_LADDER``; the layer
    maps it to whichever control the chosen model exposes, or drops it when the
    model has none. ``temperature`` is honoured only by models that actually
    act on it.
    """

    effort: Optional[str] = None
    max_output_tokens: Optional[int] = None
    temperature: Optional[float] = None


# ── What the layer changed, and why ───────────────────────────────────────────

ACTION_DROPPED = "dropped"
ACTION_TRANSLATED = "translated"


@dataclass(frozen=True)
class Adjustment:
    """One option the layer dropped or translated, with the reason.

    Story 14: the Reader should be able to see *why* an option did not survive,
    because that is how the differences between current models get taught.
    """

    option: str
    action: str
    reason: str
    requested: Any = None
    sent_as: Any = None

    def __str__(self) -> str:  # pragma: no cover - convenience for Check output
        if self.action == ACTION_DROPPED:
            return "dropped {}={!r}: {}".format(self.option, self.requested, self.reason)
        return "translated {}={!r} to {!r}: {}".format(
            self.option, self.requested, self.sent_as, self.reason
        )


# ── The outgoing request and the normalized response ──────────────────────────


@dataclass(frozen=True)
class ProviderRequest:
    """Exactly what would go over the wire. The seam the Checks assert on."""

    surface: str
    model: str
    path: str
    body: Dict[str, Any]
    base_url: Optional[str] = None

    def fingerprint(self) -> str:
        """A stable string identifying this request, for record/replay matching."""
        return json.dumps(
            {
                "surface": str(getattr(self.surface, "value", self.surface)),
                "model": self.model,
                "path": self.path,
                "body": self.body,
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )


@dataclass(frozen=True)
class Usage:
    """Tokens one call used, counted the same way for every provider.

    ``input_tokens`` is every input token, including the ones read from or
    written to the prompt cache; ``cached_input_tokens`` and
    ``cache_write_input_tokens`` say how many of them were. Vendors disagree:
    Anthropic reports cache tokens beside ``input_tokens``, OpenAI and Google
    inside it. The surfaces translate, so cost is computed one way.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_input_tokens: int = 0


@dataclass(frozen=True)
class Response:
    """The normalized answer: text, tool calls, usage, stop reason."""

    model: str
    text: str
    tool_calls: Tuple[ToolCall, ...] = ()
    usage: Usage = field(default_factory=Usage)
    stop_reason: str = STOP_END_TURN
    cost_usd: float = 0.0
    adjustments: Tuple[Adjustment, ...] = ()
    raw: Optional[Mapping[str, Any]] = None
