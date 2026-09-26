"""Transports: the seam between "what we would send" and "sending it".

Everything above this line is pure translation, which is why the Checks can
assert on a ProviderRequest without a key and without a network.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from .registry import ApiSurface
from .types import ProviderRequest

try:  # pragma: no cover - typing only
    from typing import Protocol
except ImportError:  # pragma: no cover
    Protocol = object  # type: ignore[assignment]


class Transport(Protocol):  # pragma: no cover - structural type
    """Send a built request and return the provider's raw JSON payload."""

    def send(self, request: ProviderRequest) -> Dict[str, Any]:
        ...


class TransportError(RuntimeError):
    """The request could not be delivered, or came back as an error."""


#: Per-provider credential env var for OpenAI-compatible servers.
_COMPATIBLE_KEYS = {
    "deepseek": "DEEPSEEK_API_KEY",
    "xai": "XAI_API_KEY",
    "qwen": "QWEN_API_KEY",
}


class HttpTransport:
    """Calls the real API over HTTPS. Never used by the test suite.

    It speaks raw JSON rather than a vendor SDK on purpose: the request is the
    seam the Checks assert on and the thing the replay transport records, and
    an SDK would hide it.
    """

    def __init__(self, *, provider: str = "", timeout: float = 120.0) -> None:
        self._provider = provider
        self._timeout = timeout

    def _headers(self, request: ProviderRequest) -> Dict[str, str]:
        headers = {"content-type": "application/json"}
        surface = request.surface
        if surface == ApiSurface.ANTHROPIC_MESSAGES:
            headers["x-api-key"] = _require_key("ANTHROPIC_API_KEY")
            headers["anthropic-version"] = "2023-06-01"
        elif surface == ApiSurface.OPENAI_RESPONSES:
            headers["authorization"] = "Bearer " + _require_key("OPENAI_API_KEY")
        elif surface == ApiSurface.GOOGLE_GEMINI:
            headers["x-goog-api-key"] = _require_key("GEMINI_API_KEY")
        elif self._provider == "local":
            # A local server (Ollama, vLLM) needs no key. Send one only if set.
            key = os.environ.get("LOCAL_LLM_API_KEY")
            if key:
                headers["authorization"] = "Bearer " + key
        else:
            env = _COMPATIBLE_KEYS.get(self._provider, "OPENAI_API_KEY")
            headers["authorization"] = "Bearer " + _require_key(env)
        return headers

    def send(self, request: ProviderRequest) -> Dict[str, Any]:
        if not request.base_url:
            raise TransportError(
                "No base URL for {}. Set one on the registry entry.".format(request.model)
            )
        url = request.base_url.rstrip("/") + request.path
        payload = json.dumps(request.body).encode("utf-8")
        http_request = urllib.request.Request(
            url, data=payload, headers=self._headers(request), method="POST"
        )
        try:
            with urllib.request.urlopen(http_request, timeout=self._timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:  # pragma: no cover - live only
            detail = error.read().decode("utf-8", "replace")
            raise TransportError(
                "{} returned {}: {}".format(request.model, error.code, detail)
            ) from error
        except urllib.error.URLError as error:  # pragma: no cover - live only
            raise TransportError(
                "could not reach {}: {}".format(url, error.reason)
            ) from error


#: Other names a Reader's existing `.env` may use for the same key. The vendor's own
#: name is read first; `GROK_API_KEY` is what `.env.example` shipped before this layer.
_KEY_ALIASES = {
    "XAI_API_KEY": ("GROK_API_KEY",),
    "GEMINI_API_KEY": ("GOOGLE_API_KEY",),
}


def _require_key(env_var: str) -> str:
    names = (env_var,) + _KEY_ALIASES.get(env_var, ())
    for name in names:
        key: Optional[str] = os.environ.get(name)
        if key:
            return key
    raise TransportError(
        "{} is not set. Offline Checks need no key; a live run does.".format(" or ".join(names))
    )
