"""Transports: the seam between "what we would send" and "sending it".

Everything above this line is pure translation, which is why the Checks can
assert on a ProviderRequest without a key and without a network.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict

from . import credentials
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


#: The provider whose key a surface's own vendor API takes, when the transport
#: was built without one. An OpenAI-compatible surface belongs to whoever serves it.
_SURFACE_PROVIDER = {
    ApiSurface.ANTHROPIC_MESSAGES: "anthropic",
    ApiSurface.OPENAI_RESPONSES: "openai",
    ApiSurface.GOOGLE_GEMINI: "google",
    ApiSurface.OPENAI_CHAT_COMPLETIONS: "openai",
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
        provider = self._provider or _SURFACE_PROVIDER[surface]
        if provider in credentials.OPTIONAL_KEY_PROVIDERS:
            # A local server (Ollama, vLLM) needs no key. Send one only if set.
            key = credentials.find_key(provider)
        else:
            try:
                key = credentials.require_key(provider)
            except credentials.MissingKeyError as error:
                raise TransportError(str(error)) from None
        if surface == ApiSurface.ANTHROPIC_MESSAGES:
            headers["anthropic-version"] = "2023-06-01"
        if key is None:
            return headers
        if surface == ApiSurface.ANTHROPIC_MESSAGES:
            headers["x-api-key"] = key
        elif surface == ApiSurface.GOOGLE_GEMINI:
            headers["x-goog-api-key"] = key
        else:
            headers["authorization"] = "Bearer " + key
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
