"""Seam 2 Checks: record once, replay free, fail loudly when the request moves.

    "Record mode writes a recording, replay mode returns it, and a changed
     request raises a clear mismatch error."

Stories 53 and 54: Offline Checks replay recorded model responses, and stale
recordings must not pass silently.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from provider import load_registry  # noqa: E402
from provider.call import CallOptions, complete  # noqa: E402
from provider.replay import (  # noqa: E402
    RecordingTransport,
    ReplayMismatchError,
    ReplayTransport,
)
from provider.testing import StubTransport  # noqa: E402
from provider.types import Message  # noqa: E402

REGISTRY = load_registry()
MODEL = "claude-sonnet-5"


def _ask(transport, *, text="What is Acme's refund window?", **option_kwargs):
    return complete(
        model=MODEL,
        messages=[Message.user(text)],
        options=CallOptions(**option_kwargs),
        transport=transport,
        registry=REGISTRY,
    )


def test_record_mode_writes_a_recording(tmp_path):
    path = tmp_path / "cassette.json"
    live = StubTransport(
        [
            {
                "content": [{"type": "text", "text": "Fourteen days."}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 42, "output_tokens": 7},
            }
        ]
    )
    recorder = RecordingTransport(live, path)
    response = _ask(recorder)
    recorder.save()

    assert response.text == "Fourteen days."
    assert path.exists()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert len(saved["entries"]) == 1
    assert saved["entries"][0]["request"]["model"] == MODEL


def test_replay_mode_returns_the_recording_without_a_transport(tmp_path):
    path = tmp_path / "cassette.json"
    live = StubTransport(
        [
            {
                "content": [{"type": "text", "text": "Fourteen days."}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 42, "output_tokens": 7},
            }
        ]
    )
    recorder = RecordingTransport(live, path)
    _ask(recorder)
    recorder.save()

    replayed = _ask(ReplayTransport(path))
    assert replayed.text == "Fourteen days."
    assert replayed.usage.input_tokens == 42
    assert replayed.cost_usd == pytest.approx(
        REGISTRY.get(MODEL).cost_usd(input_tokens=42, output_tokens=7)
    )


def test_the_same_request_can_be_replayed_more_than_once(tmp_path):
    path = tmp_path / "cassette.json"
    recorder = RecordingTransport(StubTransport(), path)
    _ask(recorder)
    recorder.save()

    replay = ReplayTransport(path)
    assert _ask(replay).text == _ask(replay).text


def test_a_changed_request_raises_a_clear_mismatch_error(tmp_path):
    path = tmp_path / "cassette.json"
    recorder = RecordingTransport(StubTransport(), path)
    _ask(recorder)
    recorder.save()

    with pytest.raises(ReplayMismatchError) as excinfo:
        _ask(ReplayTransport(path), text="What is Acme's proration policy?")

    message = str(excinfo.value)
    assert MODEL in message
    assert str(path) in message
    # It has to say what moved, not just that something did.
    assert "re-record" in message.lower()
    assert "body" in message.lower()


def test_an_option_change_that_alters_the_request_also_mismatches(tmp_path):
    """A capability change is exactly the silent drift recordings must catch."""
    path = tmp_path / "cassette.json"
    recorder = RecordingTransport(StubTransport(), path)
    _ask(recorder)
    recorder.save()

    with pytest.raises(ReplayMismatchError):
        _ask(ReplayTransport(path), effort="max")


def test_a_dropped_option_does_not_break_a_recording(tmp_path):
    """Temperature never reaches claude-sonnet-5, so asking for one changes nothing."""
    path = tmp_path / "cassette.json"
    recorder = RecordingTransport(StubTransport(), path)
    _ask(recorder)
    recorder.save()

    replayed = _ask(ReplayTransport(path), temperature=0.7)
    assert replayed.text == "ok"
    assert [a.option for a in replayed.adjustments] == ["temperature"]


def test_replaying_against_a_missing_recording_file_says_so(tmp_path):
    with pytest.raises(FileNotFoundError) as excinfo:
        ReplayTransport(tmp_path / "nothing-here.json")
    assert "nothing-here.json" in str(excinfo.value)


def test_a_recording_round_trips_every_surface(tmp_path):
    path = tmp_path / "all-surfaces.json"
    recorder = RecordingTransport(StubTransport(), path)
    for spec in REGISTRY.models:
        complete(
            model=spec.model_id,
            messages=[Message.user("hello")],
            transport=recorder,
            registry=REGISTRY,
        )
    recorder.save()

    replay = ReplayTransport(path)
    for spec in REGISTRY.models:
        response = complete(
            model=spec.model_id,
            messages=[Message.user("hello")],
            transport=replay,
            registry=REGISTRY,
        )
        assert response.text == "ok", spec.model_id
