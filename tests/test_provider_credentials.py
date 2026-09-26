"""Credential env vars: the vendor's own name wins, the old course names still work.

`.env.example` shipped `GROK_API_KEY` for xAI before the provider layer adopted the
vendor's `XAI_API_KEY`; Google documents both `GEMINI_API_KEY` and `GOOGLE_API_KEY`.
A Reader's existing `.env` must keep working, so the alias is read when the primary
name is unset. No network: headers are built, nothing is sent.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.transport import TransportError, _require_key  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("XAI_API_KEY", "GROK_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_old_grok_name_still_works(monkeypatch):
    monkeypatch.setenv("GROK_API_KEY", "old-name")
    assert _require_key("XAI_API_KEY") == "old-name"


def test_vendor_name_wins_over_alias(monkeypatch):
    monkeypatch.setenv("GROK_API_KEY", "old-name")
    monkeypatch.setenv("XAI_API_KEY", "vendor-name")
    assert _require_key("XAI_API_KEY") == "vendor-name"


def test_google_api_key_is_accepted_for_gemini(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "g")
    assert _require_key("GEMINI_API_KEY") == "g"


def test_missing_key_names_every_accepted_variable():
    with pytest.raises(TransportError) as err:
        _require_key("XAI_API_KEY")
    assert "XAI_API_KEY" in str(err.value) and "GROK_API_KEY" in str(err.value)
