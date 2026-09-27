"""The one-line call Appendix scripts and notebooks use: ``llm.ask``.

Appendix code used to call ``shared/provider.py``'s ``chat()``, which sent
``temperature=0.7`` to every model. ``ask`` is the same one-liner routed through
the provider layer, so a Reader's configured model never receives an option it
rejects. Tested on the outgoing request through the stub transport, like every
other Seam 2 test.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm import ask, credentials, default_model, load_registry  # noqa: E402
from llm.testing import StubTransport  # noqa: E402

REGISTRY = load_registry()
ENV_EXAMPLE = Path(__file__).parent.parent / ".env.example"

CREDENTIAL_VARS = credentials.ENV_VARS


@pytest.fixture
def no_keys(monkeypatch):
    for name in CREDENTIAL_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_ask_returns_the_answer_text(no_keys):
    transport = StubTransport()

    assert ask("What is 2+2?", model="claude-sonnet-5", transport=transport) == "ok"


def test_ask_never_sends_temperature_to_a_model_that_rejects_it(no_keys):
    transport = StubTransport()

    ask("Be creative.", model="claude-sonnet-5", temperature=0.7, transport=transport)

    assert "temperature" not in transport.last_request.body


def test_ask_sends_temperature_to_a_model_that_accepts_it(no_keys):
    transport = StubTransport()

    ask("Be creative.", model="claude-haiku-4-5", temperature=0.7, transport=transport)

    assert transport.last_request.body["temperature"] == 0.7


def test_ask_puts_the_system_prompt_where_the_provider_expects_it(no_keys):
    transport = StubTransport()

    ask("Hi", system="You are terse.", model="claude-sonnet-5", transport=transport)

    assert transport.last_request.body["system"] == "You are terse."


def test_ask_sends_openai_models_to_the_responses_api(no_keys):
    transport = StubTransport()

    ask("Hi", model="gpt-6-luna", transport=transport)

    assert transport.last_request.path == "/v1/responses"


def test_llm_model_env_var_picks_the_model(no_keys):
    no_keys.setenv("LLM_MODEL", "gemini-3.8-flash")

    assert default_model() == "gemini-3.8-flash"


def test_an_unknown_llm_model_is_refused_with_the_known_ids(no_keys):
    no_keys.setenv("LLM_MODEL", "gpt-4o-mini")  # historical-exception: a retired ID

    with pytest.raises(KeyError, match="claude-sonnet-5"):
        default_model()


@pytest.mark.parametrize(
    "key, expected",
    [
        ("ANTHROPIC_API_KEY", "claude-haiku-4-5"),
        ("OPENAI_API_KEY", "gpt-6-luna"),
        ("GEMINI_API_KEY", "gemini-3.8-flash"),
        ("GOOGLE_API_KEY", "gemini-3.8-flash"),
        ("DEEPSEEK_API_KEY", "deepseek-flash"),
        ("GROK_API_KEY", "grok-4.7"),
    ],
)
def test_the_key_you_have_picks_that_providers_cheapest_model(no_keys, key, expected):
    no_keys.setenv(key, "test-key")

    assert default_model() == expected


def test_llm_provider_env_var_overrides_key_detection(no_keys):
    no_keys.setenv("ANTHROPIC_API_KEY", "test-key")
    no_keys.setenv("LLM_PROVIDER", "openai")

    assert REGISTRY.get(default_model()).provider == "openai"


def test_no_key_and_no_model_is_a_clear_error(no_keys):
    with pytest.raises(RuntimeError, match="LLM_MODEL"):
        default_model()


# ── The documented setup: cp .env.example .env ───────────────────────────────


def _env_example():
    """The root .env.example as python-dotenv reads it, which is how the
    Appendix notebooks load a Reader's .env."""
    from dotenv import dotenv_values

    return dict(dotenv_values(ENV_EXAMPLE))


@pytest.mark.parametrize(
    "key, provider",
    [
        ("OPENAI_API_KEY", "openai"),  # the one SETUP.md asks for
        ("ANTHROPIC_API_KEY", "anthropic"),
        ("DEEPSEEK_API_KEY", "deepseek"),
        ("XAI_API_KEY", "xai"),
        ("GEMINI_API_KEY", "google"),
    ],
)
def test_the_key_pasted_over_the_template_picks_the_model_not_a_placeholder(key, provider):
    env = _env_example()
    assert key in env, "{} is no longer in .env.example".format(key)
    env[key] = "a-real-key-for-{}".format(provider)

    assert REGISTRY.get(default_model(environ=env)).provider == provider


def test_a_fresh_copy_of_the_template_configures_no_provider():
    with pytest.raises(RuntimeError, match="placeholder"):
        default_model(environ=_env_example())


# ── A local server: LLM_PROVIDER=ollama and OPENAI_BASE_URL ───────────────────
#
# Before the provider layer, `LLM_PROVIDER=ollama` plus
# `OPENAI_BASE_URL=http://localhost:11434/v1` ran the Appendix on a local model.
# The same settings still do, through the zero-priced local-server spec.

OLLAMA = "http://localhost:11434/v1"


def test_llm_provider_ollama_with_a_base_url_and_model_runs_locally(no_keys):
    no_keys.setenv("LLM_PROVIDER", "ollama")
    no_keys.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
    no_keys.setenv("LLM_MODEL", "llama3.2")
    transport = StubTransport()

    assert default_model() == "llama3.2"
    assert ask("Hi", transport=transport) == "ok"
    request = transport.last_request
    assert request.model == "llama3.2"
    assert request.base_url == "http://127.0.0.1:8000/v1"
    assert request.path == "/chat/completions"


def test_llm_provider_ollama_alone_uses_ollamas_default_address(no_keys):
    no_keys.setenv("LLM_PROVIDER", "ollama")
    transport = StubTransport()

    ask("Hi", transport=transport)

    assert transport.last_request.base_url == OLLAMA
    assert transport.last_request.model == default_model()


def test_llm_provider_local_is_the_same_as_ollama(no_keys):
    no_keys.setenv("LLM_PROVIDER", "local")
    no_keys.setenv("LLM_MODEL", "qwen3:8b")

    assert default_model() == "qwen3:8b"


def test_openai_base_url_alone_selects_the_local_server(no_keys):
    no_keys.setenv("OPENAI_BASE_URL", OLLAMA)
    transport = StubTransport()

    ask("Hi", transport=transport)

    assert transport.last_request.base_url == OLLAMA
    assert REGISTRY.get("claude-sonnet-5")  # the registry itself is untouched


def test_a_local_model_is_sent_temperature(no_keys):
    no_keys.setenv("OPENAI_BASE_URL", OLLAMA)
    no_keys.setenv("LLM_MODEL", "llama3.2")
    transport = StubTransport()

    ask("Be creative.", temperature=0.7, transport=transport)

    assert transport.last_request.body["temperature"] == 0.7


def test_a_registry_model_id_in_llm_model_beats_the_base_url(no_keys):
    no_keys.setenv("OPENAI_BASE_URL", OLLAMA)
    no_keys.setenv("LLM_MODEL", "claude-sonnet-5")

    assert default_model() == "claude-sonnet-5"


def test_the_appendix_registry_pattern_finds_the_local_model(no_keys):
    """`REGISTRY = configured_registry(); REGISTRY.get(default_model(REGISTRY))`."""
    from llm import configured_registry

    no_keys.setenv("LLM_PROVIDER", "ollama")
    no_keys.setenv("LLM_MODEL", "llama3.2")
    registry = configured_registry()

    spec = registry.get(default_model(registry))

    assert spec.provider == "local"
    assert spec.base_url == OLLAMA
    assert spec.input_price_per_mtok == 0


def test_configured_registry_is_the_plain_registry_without_a_local_server(no_keys):
    from llm import configured_registry

    assert configured_registry().ids() == REGISTRY.ids()


def test_ask_on_a_local_server_builds_a_transport_that_needs_no_key(no_keys, monkeypatch):
    import llm.transport

    built = []

    class Recording(StubTransport):
        def __init__(self, *, provider="", timeout=120.0):
            super().__init__()
            built.append(provider)

    monkeypatch.setattr(llm.transport, "HttpTransport", Recording)
    no_keys.setenv("LLM_PROVIDER", "ollama")

    assert ask("Hi") == "ok"
    assert built == ["local"]
