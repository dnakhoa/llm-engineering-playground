"""The model registry: one data file of current models and what they accept.

A Drop means editing ``models.json``. Nothing else in the course hard-codes a
model ID, a price or a capability.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Sequence, Tuple

REGISTRY_PATH = Path(__file__).with_name("models.json")

#: Every effort level any supported provider exposes, weakest first. A model's
#: own levels are a subset of this ladder, which is what makes "nearest level"
#: a well-defined translation.
EFFORT_LADDER: Tuple[str, ...] = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)


class ApiSurface(str, Enum):
    """The wire shape a model is called with."""

    ANTHROPIC_MESSAGES = "anthropic_messages"
    OPENAI_RESPONSES = "openai_responses"
    GOOGLE_GEMINI = "google_gemini"
    OPENAI_CHAT_COMPLETIONS = "openai_chat_completions"


@dataclass(frozen=True)
class ModelSpec:
    """One registry entry: a model's identity, prices and capabilities."""

    model_id: str
    provider: str
    api_surface: ApiSurface
    input_price_per_mtok: float
    output_price_per_mtok: float
    context_window: int
    accepts_sampling_params: bool
    effort_levels: Tuple[str, ...]
    source: str
    verified_on: str
    max_output_tokens: Optional[int] = None
    #: Price of an input token served from the prompt cache. ``None`` bills it
    #: at the full input price, which overstates cost rather than hiding it.
    cache_read_price_per_mtok: Optional[float] = None
    #: Price of an input token written into the prompt cache. ``None`` means the
    #: vendor lists no separate write rate, so it bills at the input price.
    cache_write_price_per_mtok: Optional[float] = None
    cache_price_source: str = ""
    default_effort: Optional[str] = None
    base_url: Optional[str] = None
    #: Why this model does not act on sampling parameters. Set whenever
    #: ``accepts_sampling_params`` is false, so the Reader is told why the
    #: layer dropped their temperature rather than just that it did.
    sampling_note: str = ""
    #: Effort levels at which a model that otherwise refuses sampling parameters
    #: does accept them. GPT-6 takes a temperature only at effort ``none``.
    sampling_effort_levels: Tuple[str, ...] = ()
    notes: str = ""

    @property
    def supports_effort(self) -> bool:
        return bool(self.effort_levels)

    def accepts_sampling_at(self, effort: Optional[str]) -> bool:
        """Whether a temperature is honoured when the request runs at ``effort``."""
        if self.accepts_sampling_params:
            return True
        return effort is not None and effort in self.sampling_effort_levels

    def cost_usd(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int = 0,
        cache_write_input_tokens: int = 0,
    ) -> float:
        """What this many tokens cost on this model, at registry prices.

        ``input_tokens`` counts every input token, cached and written ones
        included (``llm.types.Usage``). Cached tokens bill at the cache-read
        price and written tokens at the cache-write price; the rest at the
        input price.
        """
        read_price = self.cache_read_price_per_mtok
        write_price = self.cache_write_price_per_mtok
        uncached = max(0, input_tokens - cached_input_tokens - cache_write_input_tokens)
        return (
            uncached * self.input_price_per_mtok
            + cached_input_tokens
            * (self.input_price_per_mtok if read_price is None else read_price)
            + cache_write_input_tokens
            * (self.input_price_per_mtok if write_price is None else write_price)
            + output_tokens * self.output_price_per_mtok
        ) / 1_000_000


@dataclass(frozen=True)
class Registry:
    models: Tuple[ModelSpec, ...]

    def get(self, model_id: str) -> ModelSpec:
        for spec in self.models:
            if spec.model_id == model_id:
                return spec
        known = ", ".join(spec.model_id for spec in self.models)
        raise KeyError(
            "{!r} is not in the model registry. Known models: {}. "
            "Add it to llm/models.json with its vendor source and "
            "verification date.".format(model_id, known)
        )

    def ids(self) -> Tuple[str, ...]:
        return tuple(spec.model_id for spec in self.models)

    def with_model(self, spec: ModelSpec) -> "Registry":
        """This registry plus ``spec``, which replaces any entry with its ID."""
        kept = tuple(m for m in self.models if m.model_id != spec.model_id)
        return Registry(models=kept + (spec,))


#: The provider name for a local OpenAI-compatible server (Ollama, vLLM, ...).
LOCAL_PROVIDER = "local"


def local_model(model_id: str, base_url: str) -> ModelSpec:
    """A model served by a local OpenAI-compatible server, with no registry entry.

    It is priced at zero, because the Reader's own machine sends no bill, so a
    Spend Cap does not apply to it. Nothing is known about its capabilities,
    so no effort level is sent and a temperature is passed through: an
    OpenAI-compatible server accepts one.
    """
    return ModelSpec(
        model_id=model_id,
        provider=LOCAL_PROVIDER,
        api_surface=ApiSurface.OPENAI_CHAT_COMPLETIONS,
        input_price_per_mtok=0.0,
        output_price_per_mtok=0.0,
        cache_read_price_per_mtok=0.0,
        cache_write_price_per_mtok=0.0,
        context_window=0,
        accepts_sampling_params=True,
        effort_levels=(),
        source=base_url,
        verified_on="",
        base_url=base_url,
        notes="A local OpenAI-compatible server, priced at zero.",
    )


def _price(value) -> Optional[float]:
    return None if value is None else float(value)


def _spec_from_dict(entry: dict) -> ModelSpec:
    return ModelSpec(
        model_id=entry["model_id"],
        provider=entry["provider"],
        api_surface=ApiSurface(entry["api_surface"]),
        input_price_per_mtok=float(entry["input_price_per_mtok"]),
        output_price_per_mtok=float(entry["output_price_per_mtok"]),
        context_window=int(entry["context_window"]),
        accepts_sampling_params=bool(entry["accepts_sampling_params"]),
        effort_levels=tuple(entry.get("effort_levels") or ()),
        source=entry["source"],
        verified_on=entry["verified_on"],
        max_output_tokens=entry.get("max_output_tokens"),
        cache_read_price_per_mtok=_price(entry.get("cache_read_price_per_mtok")),
        cache_write_price_per_mtok=_price(entry.get("cache_write_price_per_mtok")),
        cache_price_source=entry.get("cache_price_source", ""),
        default_effort=entry.get("default_effort"),
        base_url=entry.get("base_url"),
        sampling_note=entry.get("sampling_note", ""),
        sampling_effort_levels=tuple(entry.get("sampling_effort_levels") or ()),
        notes=entry.get("notes", ""),
    )


def load_registry(path: Optional[Path] = None) -> Registry:
    """Read the registry data file."""
    data = json.loads(Path(path or REGISTRY_PATH).read_text(encoding="utf-8"))
    entries: Sequence[dict] = data["models"]
    return Registry(models=tuple(_spec_from_dict(entry) for entry in entries))
