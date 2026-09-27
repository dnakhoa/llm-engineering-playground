"""Credentials: one resolver decides which key a provider uses, and what counts as a key.

`llm/credentials.py` owns the provider → env-var table and the one placeholder rule.
`default_model()` (which provider has a key?) and `HttpTransport` (which key goes in
the header?) both ask it, so they cannot disagree. They used to: after
`cp .env.example .env`, the template's `XAI_API_KEY=...` sat next to the Reader's
real `GROK_API_KEY`; `default_model()` skipped the placeholder and chose xAI, and the
transport then sent `Authorization: Bearer ...`.

The vendor's own name is read first; the names older `.env` files used still work.
No network: headers are built, nothing is sent.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm import credentials, default_model, load_registry  # noqa: E402
from llm.call import build_request  # noqa: E402
from llm.registry import local_model  # noqa: E402
from llm.transport import HttpTransport, TransportError  # noqa: E402
from llm.types import Message  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_EXAMPLE = REPO_ROOT / ".env.example"
REGISTRY = load_registry()

ALL_NAMES = credentials.ENV_VARS


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ALL_NAMES:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _template():
    """The root .env.example exactly as python-dotenv reads it."""
    from dotenv import dotenv_values

    return {k: v for k, v in dotenv_values(ENV_EXAMPLE).items() if v is not None}


def _load(monkeypatch, env):
    for name, value in env.items():
        monkeypatch.setenv(name, value)


def _headers_for(model_id):
    spec = REGISTRY.get(model_id)
    request = build_request(spec, messages=[Message.user("hi")])
    return HttpTransport(provider=spec.provider)._headers(request)


def _sent_key(headers):
    for name in ("x-api-key", "x-goog-api-key"):
        if name in headers:
            return headers[name]
    return headers["authorization"][len("Bearer "):]


# ── The rule ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("value", [None, "", "   ", "...", "sk-...", "sk-ant-...", " sk-... "])
def test_empty_and_template_placeholders_are_unset(value):
    assert credentials.is_unset(value)


@pytest.mark.parametrize("value", ["sk-real", "a.b.c", "xai-123"])
def test_anything_else_is_a_key(value):
    assert not credentials.is_unset(value)


def test_the_vendor_name_wins_over_the_alias():
    env = {"GROK_API_KEY": "old-name", "XAI_API_KEY": "vendor-name"}

    assert credentials.find_key("xai", env) == "vendor-name"


def test_the_old_grok_name_still_works():
    assert credentials.find_key("xai", {"GROK_API_KEY": "old-name"}) == "old-name"


def test_google_api_key_is_accepted_for_gemini():
    assert credentials.find_key("google", {"GOOGLE_API_KEY": "g"}) == "g"


def test_a_placeholder_under_the_vendor_name_falls_through_to_the_alias():
    env = {"XAI_API_KEY": "...", "GROK_API_KEY": "real-grok"}

    assert credentials.find_key("xai", env) == "real-grok"


def test_old_provider_names_are_understood():
    assert credentials.normalize_provider("grok") == "xai"
    assert credentials.normalize_provider("Gemini") == "google"
    assert credentials.normalize_provider("claude") == "anthropic"
    assert credentials.normalize_provider("ollama") == "local"


# ── The error ─────────────────────────────────────────────────────────────────


def test_a_missing_key_names_every_variable_tried():
    with pytest.raises(credentials.MissingKeyError) as err:
        credentials.require_key("xai", {})

    message = str(err.value)
    assert "XAI_API_KEY" in message and "GROK_API_KEY" in message


def test_a_missing_key_says_a_placeholder_was_skipped():
    with pytest.raises(credentials.MissingKeyError) as err:
        credentials.require_key("xai", {"XAI_API_KEY": "..."})

    message = str(err.value)
    assert "placeholder" in message
    assert "XAI_API_KEY" in message and "GROK_API_KEY" in message


def test_the_transport_reports_a_missing_key_as_a_transport_error(monkeypatch):
    with pytest.raises(TransportError, match="GROK_API_KEY"):
        _headers_for("grok-4.7")


# ── The documented setup: cp .env.example .env, then one real key ─────────────


def test_after_copying_the_template_the_grok_alias_key_is_the_one_sent(monkeypatch):
    env = _template()
    assert env.get("XAI_API_KEY") == "...", "the template no longer ships XAI_API_KEY=..."
    env["GROK_API_KEY"] = "real-grok-key"
    _load(monkeypatch, env)

    assert REGISTRY.get(default_model()).provider == "xai"
    assert _headers_for(default_model())["authorization"] == "Bearer real-grok-key"


def test_after_copying_the_template_the_google_alias_key_is_the_one_sent(monkeypatch):
    env = _template()
    assert env.get("GEMINI_API_KEY") == "...", "the template no longer ships GEMINI_API_KEY=..."
    env["GOOGLE_API_KEY"] = "real-google-key"
    _load(monkeypatch, env)

    assert REGISTRY.get(default_model()).provider == "google"
    assert _headers_for(default_model())["x-goog-api-key"] == "real-google-key"


PROVIDER_VARS = [
    (provider, name)
    for provider, names in credentials.PROVIDER_KEYS
    if provider not in credentials.OPTIONAL_KEY_PROVIDERS
    and any(spec.provider == provider for spec in REGISTRY.models)
    for name in names
]


@pytest.mark.parametrize("provider, name", PROVIDER_VARS, ids=[n for _, n in PROVIDER_VARS])
def test_the_model_chosen_and_the_key_sent_come_from_the_same_variable(
    monkeypatch, provider, name
):
    env = _template()
    env[name] = "real-key-in-" + name
    _load(monkeypatch, env)

    model = default_model()

    assert REGISTRY.get(model).provider == provider
    assert _sent_key(_headers_for(model)) == "real-key-in-" + name


# ── A local server ────────────────────────────────────────────────────────────


def _local_headers():
    spec = local_model("qwen3:8b", "http://localhost:11434/v1")
    request = build_request(spec, messages=[Message.user("hi")])
    return HttpTransport(provider=spec.provider)._headers(request)


def test_a_local_server_needs_no_key():
    assert "authorization" not in _local_headers()


def test_a_local_server_gets_its_key_when_one_is_set(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_API_KEY", "local-secret")

    assert _local_headers()["authorization"] == "Bearer local-secret"


def test_a_local_placeholder_key_is_not_sent(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_API_KEY", "...")

    assert "authorization" not in _local_headers()


def test_a_local_server_never_receives_the_openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-openai")

    assert "authorization" not in _local_headers()


# ── One table ─────────────────────────────────────────────────────────────────


def test_no_other_module_in_the_layer_names_a_key_variable():
    """The env-var table lives in llm/credentials.py only, so it cannot drift."""
    offenders = [
        path.name
        for path in (REPO_ROOT / "llm").rglob("*.py")
        if path.name != "credentials.py" and "_API_KEY" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []
