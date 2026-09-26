"""A stub transport, so the Checks can look at the request without sending it.

The Checks assert on the outgoing request. The reply only has to be well formed
enough to parse, so by default the stub answers each surface with that
surface's smallest valid shape; queue payloads to say something specific.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .registry import ApiSurface
from .types import ProviderRequest

_DEFAULT_PAYLOADS = {
    ApiSurface.ANTHROPIC_MESSAGES: {
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    },
    ApiSurface.OPENAI_RESPONSES: {
        "status": "completed",
        "output": [
            {"type": "message", "content": [{"type": "output_text", "text": "ok"}]}
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    },
    ApiSurface.OPENAI_CHAT_COMPLETIONS: {
        "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    },
    ApiSurface.GOOGLE_GEMINI: {
        "candidates": [
            {"content": {"role": "model", "parts": [{"text": "ok"}]}, "finishReason": "STOP"}
        ],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
    },
}


class StubTransport:
    """Records every request; replies from a queue, else a per-surface default."""

    def __init__(self, payloads: Optional[Sequence[Dict[str, Any]]] = None) -> None:
        self.requests: List[ProviderRequest] = []
        self._queue: List[Dict[str, Any]] = list(payloads or [])

    def send(self, request: ProviderRequest) -> Dict[str, Any]:
        self.requests.append(request)
        if self._queue:
            return self._queue.pop(0)
        return dict(_DEFAULT_PAYLOADS[request.surface])

    @property
    def last_request(self) -> ProviderRequest:
        if not self.requests:
            raise AssertionError("no request was built")
        return self.requests[-1]
