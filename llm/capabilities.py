"""Turning intent-level options into what one model will actually accept.

This is the part ADR 0002 is about. The caller says "think hard, keep it under
2000 tokens, be deterministic". Each model hears a different subset of that,
under a different name. Anything that cannot be said to a model is dropped or
translated here, and the reason is recorded rather than swallowed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .registry import EFFORT_LADDER, ModelSpec
from .types import ACTION_DROPPED, ACTION_TRANSLATED, Adjustment, CallOptions

#: Used only when the caller names no output ceiling and the surface demands one
#: (Anthropic's Messages API requires ``max_tokens``).
FALLBACK_MAX_OUTPUT_TOKENS = 4096


@dataclass(frozen=True)
class RequestPlan:
    """The options that survived, in the chosen model's own vocabulary."""

    effort: Optional[str]
    max_output_tokens: Optional[int]
    temperature: Optional[float]
    adjustments: Tuple[Adjustment, ...]

    def required_max_output_tokens(self, spec: ModelSpec) -> int:
        """For surfaces that make an output ceiling mandatory."""
        if self.max_output_tokens is not None:
            return self.max_output_tokens
        ceiling = spec.max_output_tokens or FALLBACK_MAX_OUTPUT_TOKENS
        return min(FALLBACK_MAX_OUTPUT_TOKENS, ceiling)


def _map_effort(spec: ModelSpec, requested: Optional[str]):
    if requested is None:
        return None, None

    if requested not in EFFORT_LADDER:
        raise ValueError(
            "Unknown effort {!r}. Use one of: {}.".format(
                requested, ", ".join(EFFORT_LADDER)
            )
        )

    if not spec.effort_levels:
        return None, Adjustment(
            option="effort",
            action=ACTION_DROPPED,
            reason="{} has no effort control, so the level cannot be expressed.".format(
                spec.model_id
            ),
            requested=requested,
        )

    if requested in spec.effort_levels:
        return requested, None

    # Snap to the nearest level this model does have; ties go to the lower one,
    # so a translation never quietly spends more than the caller asked for.
    wanted = EFFORT_LADDER.index(requested)
    nearest = min(
        spec.effort_levels,
        key=lambda level: (abs(EFFORT_LADDER.index(level) - wanted), EFFORT_LADDER.index(level)),
    )
    return nearest, Adjustment(
        option="effort",
        action=ACTION_TRANSLATED,
        reason="{} supports only {}.".format(
            spec.model_id, ", ".join(spec.effort_levels)
        ),
        requested=requested,
        sent_as=nearest,
    )


def _resolve_temperature(spec: ModelSpec, requested: Optional[float]):
    if requested is None:
        return None, None
    if spec.accepts_sampling_params:
        return requested, None
    return None, Adjustment(
        option="temperature",
        action=ACTION_DROPPED,
        reason="{}: {}".format(
            spec.model_id,
            spec.sampling_note or "this model does not act on sampling parameters.",
        ),
        requested=requested,
    )


def _resolve_max_output(spec: ModelSpec, requested: Optional[int]):
    if requested is None:
        return None, None
    ceiling = spec.max_output_tokens
    if ceiling is not None and requested > ceiling:
        return ceiling, Adjustment(
            option="max_output_tokens",
            action=ACTION_TRANSLATED,
            reason="{} caps output at {} tokens.".format(spec.model_id, ceiling),
            requested=requested,
            sent_as=ceiling,
        )
    return requested, None


def plan_request(spec: ModelSpec, options: Optional[CallOptions]) -> RequestPlan:
    options = options or CallOptions()
    adjustments: List[Adjustment] = []

    effort, effort_adjustment = _map_effort(spec, options.effort)
    temperature, temperature_adjustment = _resolve_temperature(spec, options.temperature)
    max_output, max_output_adjustment = _resolve_max_output(
        spec, options.max_output_tokens
    )

    for adjustment in (effort_adjustment, temperature_adjustment, max_output_adjustment):
        if adjustment is not None:
            adjustments.append(adjustment)

    return RequestPlan(
        effort=effort,
        max_output_tokens=max_output,
        temperature=temperature,
        adjustments=tuple(adjustments),
    )
