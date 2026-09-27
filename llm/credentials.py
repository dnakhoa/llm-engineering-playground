"""Credentials: which environment variable holds each provider's key, and what counts.

This module is the only place in the provider layer that names a key variable.
``default_model()`` asks it which providers have a key; ``HttpTransport`` asks it
which key to send. Asking one resolver is what keeps the two from disagreeing.

Each provider has an ordered list of names. The vendor's own name comes first;
after it come the names older ``.env`` files used (``GROK_API_KEY`` shipped in
``.env.example`` before the vendor's ``XAI_API_KEY``; Google documents both
``GEMINI_API_KEY`` and ``GOOGLE_API_KEY``). The first name holding a key wins.

One rule decides what a key is: a value that is empty, all whitespace, or ends
with ``...`` is unset. ``...`` is how ``.env.example`` writes the key it leaves
for the Reader (``sk-...``, ``sk-ant-...``), so ``cp .env.example .env`` and one
real key under any accepted name selects, and sends, that real key.

A local OpenAI-compatible server (Ollama, vLLM) needs no key. Its address comes
from ``OPENAI_BASE_URL``, read under the same rule.
"""
from __future__ import annotations

import os
from typing import List, Mapping, Optional, Tuple

#: Provider → the env vars that may hold its key, vendor name first. The order of
#: providers is the order ``default_model()`` detects them in.
PROVIDER_KEYS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("anthropic", ("ANTHROPIC_API_KEY",)),
    ("deepseek", ("DEEPSEEK_API_KEY",)),
    ("xai", ("XAI_API_KEY", "GROK_API_KEY")),
    ("openai", ("OPENAI_API_KEY",)),
    ("google", ("GEMINI_API_KEY", "GOOGLE_API_KEY")),
    ("qwen", ("QWEN_API_KEY",)),
    ("local", ("LOCAL_LLM_API_KEY",)),
)

#: Providers whose key is optional: sent when set, and never required.
OPTIONAL_KEY_PROVIDERS = frozenset({"local"})

#: Where a local OpenAI-compatible server listens, read in this order.
LOCAL_BASE_URL_VARS: Tuple[str, ...] = ("OPENAI_BASE_URL",)

#: Names older course code accepted in ``LLM_PROVIDER``.
PROVIDER_ALIASES = {"grok": "xai", "gemini": "google", "claude": "anthropic", "ollama": "local"}

#: How ``.env.example`` writes a value it leaves for the Reader.
PLACEHOLDER_SUFFIX = "..."

_KEYS = dict(PROVIDER_KEYS)

#: Every variable the layer reads to choose a model and its key: the key names,
#: the local server's address, and ``LLM_PROVIDER`` / ``LLM_MODEL``. Tests that
#: need a clean environment clear these.
ENV_VARS: Tuple[str, ...] = (
    tuple(name for _, names in PROVIDER_KEYS for name in names)
    + LOCAL_BASE_URL_VARS
    + ("LLM_PROVIDER", "LLM_MODEL")
)


class MissingKeyError(RuntimeError):
    """A provider's key is in none of the variables it may be read from."""


def is_unset(value: Optional[str]) -> bool:
    """True for no value, an empty or blank one, or a template placeholder."""
    value = (value or "").strip()
    return not value or value.endswith(PLACEHOLDER_SUFFIX)


def normalize_provider(name: str) -> str:
    """``LLM_PROVIDER`` as the registry spells it: ``grok`` → ``xai``, and so on."""
    name = (name or "").strip().lower()
    return PROVIDER_ALIASES.get(name, name)


def key_names(provider: str) -> Tuple[str, ...]:
    """The env vars that may hold ``provider``'s key, in the order they are read."""
    provider = normalize_provider(provider)
    try:
        return _KEYS[provider]
    except KeyError:
        raise KeyError(
            "no key variable is known for provider {!r}. Known providers: {}".format(
                provider, ", ".join(_KEYS)
            )
        ) from None


def _env(environ: Optional[Mapping[str, str]]) -> Mapping[str, str]:
    return os.environ if environ is None else environ


def find_key(provider: str, environ: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """``provider``'s key from the first variable that holds one, else ``None``."""
    env = _env(environ)
    for name in key_names(provider):
        value = env.get(name)
        if not is_unset(value):
            return value.strip()
    return None


def require_key(provider: str, environ: Optional[Mapping[str, str]] = None) -> str:
    """``provider``'s key, or a ``MissingKeyError`` naming every variable tried."""
    key = find_key(provider, environ)
    if key is not None:
        return key
    env = _env(environ)
    names = key_names(provider)
    placeholders = [name for name in names if (env.get(name) or "").strip()]
    message = "No {} key: tried {}.".format(provider, ", ".join(names))
    if placeholders:
        message += (
            " {} holds the .env.example placeholder (a value ending in '{}'), "
            "which is not a key; paste your real key over it.".format(
                " and ".join(placeholders), PLACEHOLDER_SUFFIX
            )
        )
    message += " Offline Checks need no key; a live run does."
    raise MissingKeyError(message)


def providers_with_keys(environ: Optional[Mapping[str, str]] = None) -> List[str]:
    """Every provider whose key is set, in detection order. Optional keys don't count."""
    return [
        provider
        for provider, _ in PROVIDER_KEYS
        if provider not in OPTIONAL_KEY_PROVIDERS and find_key(provider, environ) is not None
    ]


def local_base_url(environ: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """The local server's address from the environment, else ``None``."""
    env = _env(environ)
    for name in LOCAL_BASE_URL_VARS:
        value = env.get(name)
        if not is_unset(value):
            return value.strip()
    return None
