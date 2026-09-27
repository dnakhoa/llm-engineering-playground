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
    default_effort: Optional[str] = None
    base_url: Optional[str] = None
    #: Why this model does not act on sampling parameters. Set whenever
    #: ``accepts_sampling_params`` is false, so the Reader is told why the
    #: layer dropped their temperature rather than just that it did.
    sampling_note: str = ""
    notes: str = ""

    @property
    def supports_effort(self) -> bool:
        return bool(self.effort_levels)

    def cost_usd(self, *, input_tokens: int, output_tokens: int) -> float:
        """What this many tokens cost on this model, at registry prices."""
        return (
            input_tokens * self.input_price_per_mtok
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
        default_effort=entry.get("default_effort"),
        base_url=entry.get("base_url"),
        sampling_note=entry.get("sampling_note", ""),
        notes=entry.get("notes", ""),
    )


def load_registry(path: Optional[Path] = None) -> Registry:
    """Read the registry data file."""
    data = json.loads(Path(path or REGISTRY_PATH).read_text(encoding="utf-8"))
    entries: Sequence[dict] = data["models"]
    return Registry(models=tuple(_spec_from_dict(entry) for entry in entries))
