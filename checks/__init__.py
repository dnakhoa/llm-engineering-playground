"""Checks: runnable pass/fail assertions over a Case's Outcome.

Each Spine module owns a suite, ``checks.spine_<n>_<name>``. A suite names its
module (``MODULE``, ``TITLE``), the Cases it runs (``CASES``) and its Checks
(``CHECKS``). A Check takes an Outcome and returns a ``CheckResult`` whose
``detail`` says what it saw, so a failure explains itself.

Run them with the Checks CLI::

    python -m checks --modules 1 --agent path/to/my_agent.py:run

A module's run includes every earlier module's suite (``checks.cli``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypeVar


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


F = TypeVar("F", bound=Callable[..., CheckResult])


def live_only(check: F) -> F:
    """Mark a Check that needs a live model, such as an LLM-as-judge rubric.

    Offline, the Checks CLI reports it as skipped rather than running it:
    a recording holds the agent's calls, not the judge's.
    """
    setattr(check, "live_only", True)
    return check


def is_live_only(check: Callable[..., CheckResult]) -> bool:
    return bool(getattr(check, "live_only", False))
