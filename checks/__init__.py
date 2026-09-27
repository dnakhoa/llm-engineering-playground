"""Checks: runnable pass/fail assertions over a Case's Outcome.

Each Spine module owns a suite, ``checks.spine_<n>_<name>``. A suite names its
module (``MODULE``, ``TITLE``), the Cases it runs (``CASES``) and its Checks
(``CHECKS``). A Check takes an Outcome and returns a ``CheckResult`` whose
``detail`` says what it saw, so a failure explains itself.

Run them with the Checks CLI::

    python -m checks --modules 1 --agent path/to/my_agent.py:run

A module's run includes every earlier module's suite (``checks.cli``), and a
suite lists only its own Checks. A Check that holds on every Case
(``on_every_case``), such as one part of the Outcome's verdict, also grades
every later suite's Cases, once per Case, so a Case a later suite adds is
graded on the whole verdict without that suite listing those Checks again.
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


def on_every_case(check: F) -> F:
    """Mark a Check that holds on every Case, such as one part of the verdict.

    The Checks CLI grades every later suite's Cases on it too, once each. The
    verdict is what ``Outcome.resolved`` is read from, so a Case a later suite
    adds, which the marking suite does not run, is still failed when it is not
    resolved.
    """
    setattr(check, "on_every_case", True)
    return check


def is_on_every_case(check: Callable[..., CheckResult]) -> bool:
    return bool(getattr(check, "on_every_case", False))


def needs_trace(check: F) -> F:
    """Mark a Check that reads the Outcome's OpenTelemetry trace.

    Only such a Check needs ``opentelemetry-sdk``. Without it, the Checks CLI
    will not start a run that includes one, and says how to install it, rather
    than fail every trace Check on an empty trace.
    """
    setattr(check, "needs_trace", True)
    return check


def is_needs_trace(check: Callable[..., CheckResult]) -> bool:
    return bool(getattr(check, "needs_trace", False))
