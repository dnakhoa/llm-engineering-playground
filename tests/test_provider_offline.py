"""Check: the provider layer needs no network and no keys.

    "The whole test suite runs with no network or keys."

Story 61 and the spec's Testing Decisions. This proves it for the provider
layer rather than asserting it: sockets are made to explode and every
credential variable is removed, then every registry model is exercised through
the stub transport and through a record/replay round trip.
"""
from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm import load_registry  # noqa: E402
from llm.call import CallOptions, complete  # noqa: E402
from llm.replay import RecordingTransport, ReplayTransport  # noqa: E402
from llm.testing import StubTransport  # noqa: E402
from llm.transport import HttpTransport, TransportError  # noqa: E402
from llm.types import Message, ToolSpec  # noqa: E402

REGISTRY = load_registry()

CREDENTIAL_VARS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "DEEPSEEK_API_KEY",
    "XAI_API_KEY",
    "GROK_API_KEY",
    "QWEN_API_KEY",
    "LLM_PROVIDER",
    "LLM_MODEL",
)

LOOK_UP_ACCOUNT = ToolSpec(
    name="look_up_account",
    description="Read the acting customer's account.",
    input_schema={"type": "object", "properties": {}, "additionalProperties": False},
)


@pytest.fixture
def grounded(monkeypatch):
    """No credentials in the environment, and any socket use is an error."""
    for name in CREDENTIAL_VARS:
        monkeypatch.delenv(name, raising=False)

    def no_sockets(*args, **kwargs):
        raise AssertionError("the provider layer tried to open a socket")

    monkeypatch.setattr(socket, "socket", no_sockets)
    monkeypatch.setattr(socket, "create_connection", no_sockets)
    return None


@pytest.mark.parametrize("spec", REGISTRY.models, ids=lambda s: s.model_id)
def test_every_registry_model_answers_offline(grounded, spec, tmp_path):
    messages = [Message.user("What is Acme's refund window?")]
    options = CallOptions(effort="high", temperature=0.2, max_output_tokens=512)

    def ask(transport):
        return complete(
            model=spec.model_id,
            messages=messages,
            tools=[LOOK_UP_ACCOUNT],
            system="You are Acme Notes support.",
            options=options,
            transport=transport,
            registry=REGISTRY,
        )

    live = ask(StubTransport())
    assert live.text == "ok"

    cassette = tmp_path / "{}.json".format(spec.model_id)
    recorder = RecordingTransport(StubTransport(), cassette)
    ask(recorder)
    recorder.save()

    replayed = ask(ReplayTransport(cassette))
    assert replayed.text == live.text
    assert replayed.adjustments == live.adjustments


def test_the_live_transport_refuses_rather_than_running_keyless(grounded):
    """A live run without a key fails with a sentence, not a stack trace."""
    spec = REGISTRY.get("claude-sonnet-5")
    with pytest.raises(TransportError) as excinfo:
        complete(
            model=spec.model_id,
            messages=[Message.user("hello")],
            transport=HttpTransport(provider=spec.provider),
            registry=REGISTRY,
        )
    assert "ANTHROPIC_API_KEY" in str(excinfo.value)
