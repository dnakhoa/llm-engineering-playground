"""Record a live run once; replay it for free, for ever, or fail loudly.

Offline Checks need no API key because they replay recorded model responses.
That only stays honest if a recording refuses to answer a request it never saw:
a silently-passing stale recording is worse than no recording, because it
reports a green suite for behaviour nobody exercised (story 54).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .types import ProviderRequest

RECORDING_VERSION = 1


class ReplayMismatchError(LookupError):
    """A request arrived that the recording does not hold."""


def _key(request: ProviderRequest) -> str:
    return hashlib.sha256(request.fingerprint().encode("utf-8")).hexdigest()[:32]


def _request_summary(request: ProviderRequest) -> Dict[str, Any]:
    return {
        "surface": str(getattr(request.surface, "value", request.surface)),
        "model": request.model,
        "path": request.path,
        "body": request.body,
    }


class RecordingTransport:
    """Wraps a live transport and writes every request/response pair down."""

    def __init__(self, inner, path: Union[str, Path]) -> None:
        self._inner = inner
        self.path = Path(path)
        self.entries: List[Dict[str, Any]] = []

    def send(self, request: ProviderRequest) -> Dict[str, Any]:
        payload = self._inner.send(request)
        self.entries.append(
            {
                "key": _key(request),
                "request": _request_summary(request),
                "response": payload,
            }
        )
        return payload

    def save(self) -> Path:
        """Write the recording. Later pairs with the same key replace earlier ones."""
        deduped: Dict[str, Dict[str, Any]] = {}
        for entry in self.entries:
            deduped[entry["key"]] = entry
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {"version": RECORDING_VERSION, "entries": list(deduped.values())},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return self.path


class ReplayTransport:
    """Answers from a recording, and refuses anything it has not seen."""

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                "No recording at {}. Record one first with RecordingTransport, "
                "or run the Checks in live mode.".format(self.path)
            )
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self._entries: Dict[str, Dict[str, Any]] = {
            entry["key"]: entry for entry in data.get("entries", [])
        }

    def send(self, request: ProviderRequest) -> Dict[str, Any]:
        entry = self._entries.get(_key(request))
        if entry is None:
            raise ReplayMismatchError(self._explain(request))
        return entry["response"]

    def _explain(self, request: ProviderRequest) -> str:
        summary = _request_summary(request)
        lines = [
            "No recording in {} matches this request for {}.".format(
                self.path, request.model
            )
        ]
        nearest = self._nearest(request)
        if nearest is None:
            lines.append(
                "The recording holds nothing for {} on {}.".format(
                    request.model, summary["path"]
                )
            )
        else:
            lines.append(
                "Closest recorded request is for {} on {}; these fields differ:".format(
                    nearest["request"]["model"], nearest["request"]["path"]
                )
            )
            for field, recorded, now in _differences(nearest["request"], summary):
                lines.append(
                    "  - {}: recorded {} but the request now has {}".format(
                        field, _brief(recorded), _brief(now)
                    )
                )
        lines.append(
            "The request changed, so the recorded response no longer answers it. "
            "Re-record this Case, or fix the change that moved the request."
        )
        return "\n".join(lines)

    def _nearest(self, request: ProviderRequest) -> Optional[Dict[str, Any]]:
        same_model = [
            entry
            for entry in self._entries.values()
            if entry["request"]["model"] == request.model
        ]
        pool = same_model or list(self._entries.values())
        if not pool:
            return None
        summary = _request_summary(request)
        return min(pool, key=lambda e: len(_differences(e["request"], summary)))


def _differences(recorded: Dict[str, Any], now: Dict[str, Any]):
    """Which top-level request fields moved, body keys itemised."""
    out = []
    for field in ("surface", "model", "path"):
        if recorded.get(field) != now.get(field):
            out.append((field, recorded.get(field), now.get(field)))
    recorded_body = recorded.get("body") or {}
    current_body = now.get("body") or {}
    for key in sorted(set(recorded_body) | set(current_body)):
        if recorded_body.get(key) != current_body.get(key):
            out.append(
                ("body.{}".format(key), recorded_body.get(key), current_body.get(key))
            )
    return out


def _brief(value: Any, limit: int = 160) -> str:
    text = json.dumps(value, sort_keys=True, default=str)
    return text if len(text) <= limit else text[: limit - 3] + "..."
