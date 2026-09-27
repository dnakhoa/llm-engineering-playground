"""``ask()``: the one-line call for Appendix scripts and notebooks.

    from llm import ask
    print(ask("What is 2+2?", system="Answer in one word."))

``ask()`` is a documented thin wrapper, kept so Appendix prose can show a model
call in one line. It is ``complete()`` with one user turn and the text handed
back, so every option still goes through the registry: a ``temperature`` the
chosen model rejects is dropped, and OpenAI models are called through the
Responses API. ``complete()`` stays the canonical call, and the one the
Flagship Agent uses; reach for it when you need tool calls, usage, cost or the
list of adjustments the layer made.

Which model answers, when none is named (``default_model()``):

1. A local OpenAI-compatible server, when ``LLM_PROVIDER`` is ``ollama`` or
   ``local``, or when ``OPENAI_BASE_URL`` is set and ``LLM_MODEL`` is not a
   registry ID. The model is ``LLM_MODEL`` (default ``llama3.2``), served at
   ``OPENAI_BASE_URL`` (default Ollama's ``http://localhost:11434/v1``), priced
   at zero, with no registry entry needed.
2. ``LLM_MODEL``, when set. It must be a registry ID (``llm/models.json``).
3. Otherwise the cheapest registry model of ``LLM_PROVIDER``, when set.
4. Otherwise the cheapest registry model of the first provider whose API key
   is in the environment. Which variables hold a provider's key, and why a
   value ending in ``...`` is the ``.env.example`` placeholder rather than a
   key, is ``llm/credentials.py``'s business; the transport asks it too, so
   the model chosen and the key sent always agree.

Load your ``.env`` before calling it; this module reads ``os.environ`` only.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Mapping, Optional, Sequence

from . import credentials
from .call import complete
from .registry import LOCAL_PROVIDER, ModelSpec, Registry, load_registry, local_model
from .types import CallOptions, Message


@lru_cache(maxsize=1)
def _default_registry() -> Registry:
    return load_registry()


#: The model a local server is asked for when ``LLM_MODEL`` names none.
DEFAULT_LOCAL_MODEL = "llama3.2"

#: Where Ollama listens by default, for ``LLM_PROVIDER=ollama`` with no base URL.
DEFAULT_LOCAL_BASE_URL = "http://localhost:11434/v1"


def _cheapest(registry: Registry, provider: str) -> ModelSpec:
    candidates = [spec for spec in registry.models if spec.provider == provider]
    if not candidates:
        known = sorted({spec.provider for spec in registry.models})
        raise KeyError(
            "no registry model for provider {!r}. Providers in llm/models.json: {}. "
            "For a local server set LLM_PROVIDER=ollama (or local) and "
            "OPENAI_BASE_URL.".format(provider, ", ".join(known))
        )
    return min(candidates, key=lambda spec: spec.input_price_per_mtok)


def _local_spec(registry: Registry, env: Mapping[str, str]) -> Optional[ModelSpec]:
    """The local-server model the environment configures, if it configures one."""
    provider = credentials.normalize_provider(env.get("LLM_PROVIDER", ""))
    base_url = credentials.local_base_url(env)
    explicit = env.get("LLM_MODEL", "").strip()
    if provider != LOCAL_PROVIDER:
        # OPENAI_BASE_URL alone means a local server, unless LLM_PROVIDER names
        # a hosted provider or LLM_MODEL names a hosted registry model.
        hosted = {spec.model_id for spec in registry.models if spec.provider != LOCAL_PROVIDER}
        if provider or not base_url or explicit in hosted:
            return None
    return local_model(explicit or DEFAULT_LOCAL_MODEL, base_url or DEFAULT_LOCAL_BASE_URL)


def resolve_model(
    registry: Optional[Registry] = None, environ: Optional[Mapping[str, str]] = None
) -> ModelSpec:
    """The spec of the model ``ask()`` uses when none is named. See the module docstring."""
    registry = registry or _default_registry()
    env = os.environ if environ is None else environ

    local = _local_spec(registry, env)
    if local is not None:
        return local

    explicit = env.get("LLM_MODEL", "").strip()
    if explicit:
        return registry.get(explicit)

    provider = credentials.normalize_provider(env.get("LLM_PROVIDER", ""))
    if provider:
        return _cheapest(registry, provider)

    served = {spec.provider for spec in registry.models}
    for name in credentials.providers_with_keys(env):
        if name in served:
            return _cheapest(registry, name)

    raise RuntimeError(
        "No model configured. Set LLM_MODEL to an ID from llm/models.json, "
        "put one provider's API key in the root .env (see .env.example), or "
        "set LLM_PROVIDER=ollama for a local server. A value ending in '...' is "
        "the template's placeholder, not a key."
    )


def default_model(
    registry: Optional[Registry] = None, environ: Optional[Mapping[str, str]] = None
) -> str:
    """The ID of the model ``ask()`` uses when none is named. See the module docstring."""
    return resolve_model(registry, environ).model_id


def configured_registry(
    registry: Optional[Registry] = None, environ: Optional[Mapping[str, str]] = None
) -> Registry:
    """The registry, plus the local-server model the environment configures, if any.

    Code that looks the default model up (``REGISTRY.get(default_model(REGISTRY))``)
    loads this instead of ``load_registry()``, so a local model is found too.
    Without a local server it is the plain registry.
    """
    registry = registry or _default_registry()
    env = os.environ if environ is None else environ
    local = _local_spec(registry, env)
    return registry if local is None else registry.with_model(local)


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
    """Send ``prompt`` (or ``messages``) to ``model`` and return the answer text.

    A thin wrapper over ``complete()`` for Appendix prose: one call, the text
    back. ``model`` defaults to ``default_model()``, which may be a local server
    the environment configures. The Flagship Agent calls ``complete()``.
    """
    if (prompt is None) == (messages is None):
        raise ValueError("pass exactly one of prompt or messages")
    registry = configured_registry(registry)
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
