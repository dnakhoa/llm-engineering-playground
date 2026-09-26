"""``ask()``: the one-line call for Appendix scripts and notebooks.

    from llm import ask
    print(ask("What is 2+2?", system="Answer in one word."))

It is ``complete()`` with one user turn and the text handed back, so every
option still goes through the registry: a ``temperature`` the chosen model
rejects is dropped, and OpenAI models are called through the Responses API.
Use ``complete()`` directly when you need tool calls, usage or the list of
adjustments the layer made.

Which model answers:

1. ``LLM_MODEL``, when set. It must be a registry ID (``llm/models.json``).
2. Otherwise the cheapest registry model of ``LLM_PROVIDER``, when set.
3. Otherwise the cheapest registry model of the first provider whose API key
   is in the environment.

Load your ``.env`` before calling it; this module reads ``os.environ`` only.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Mapping, Optional, Sequence

from .call import complete
from .registry import Registry, load_registry
from .types import CallOptions, Message


@lru_cache(maxsize=1)
def _default_registry() -> Registry:
    return load_registry()


#: Provider → the env vars that hold its key, in the order they are detected.
_PROVIDER_KEYS = (
    ("anthropic", ("ANTHROPIC_API_KEY",)),
    ("deepseek", ("DEEPSEEK_API_KEY",)),
    ("xai", ("XAI_API_KEY", "GROK_API_KEY")),
    ("openai", ("OPENAI_API_KEY",)),
    ("google", ("GEMINI_API_KEY", "GOOGLE_API_KEY")),
)

#: Names the old ``shared/provider.py`` accepted in ``LLM_PROVIDER``.
_PROVIDER_ALIASES = {"grok": "xai", "gemini": "google", "claude": "anthropic"}


def _cheapest(registry: Registry, provider: str) -> str:
    candidates = [spec for spec in registry.models if spec.provider == provider]
    if not candidates:
        known = sorted({spec.provider for spec in registry.models})
        raise KeyError(
            "no registry model for provider {!r}. Providers in llm/models.json: {}".format(
                provider, ", ".join(known)
            )
        )
    return min(candidates, key=lambda spec: spec.input_price_per_mtok).model_id


def default_model(
    registry: Optional[Registry] = None, environ: Optional[Mapping[str, str]] = None
) -> str:
    """The model ``ask()`` uses when none is named. See the module docstring."""
    registry = registry or _default_registry()
    env = os.environ if environ is None else environ

    explicit = env.get("LLM_MODEL", "").strip()
    if explicit:
        return registry.get(explicit).model_id

    provider = env.get("LLM_PROVIDER", "").strip().lower()
    if provider:
        return _cheapest(registry, _PROVIDER_ALIASES.get(provider, provider))

    for name, keys in _PROVIDER_KEYS:
        if any(env.get(key) for key in keys):
            return _cheapest(registry, name)

    raise RuntimeError(
        "No model configured. Set LLM_MODEL to an ID from llm/models.json, "
        "or put one provider's API key in the root .env (see .env.example)."
    )


def ask(
    prompt: Optional[str] = None,
    *,
    system: Optional[str] = None,
    messages: Optional[Sequence[Message]] = None,
    model: Optional[str] = None,
    effort: Optional[str] = None,
    max_output_tokens: Optional[int] = 1024,
    temperature: Optional[float] = None,
    transport=None,
    registry: Optional[Registry] = None,
) -> str:
    """Send ``prompt`` (or ``messages``) to ``model`` and return the answer text."""
    if (prompt is None) == (messages is None):
        raise ValueError("pass exactly one of prompt or messages")
    registry = registry or _default_registry()
    model_id = model or default_model(registry)
    if transport is None:
        from .transport import HttpTransport

        transport = HttpTransport(provider=registry.get(model_id).provider)

    response = complete(
        model=model_id,
        messages=list(messages) if messages is not None else [Message.user(prompt)],
        system=system,
        options=CallOptions(
            effort=effort, max_output_tokens=max_output_tokens, temperature=temperature
        ),
        transport=transport,
        registry=registry,
    )
    return response.text
