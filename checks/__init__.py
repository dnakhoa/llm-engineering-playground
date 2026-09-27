"""Checks: runnable pass/fail assertions over a Case's Outcome.

Each Spine module owns a suite, ``checks.spine_<n>_<name>``, and each suite
exposes its Checks as ``CHECKS``. A Check takes an Outcome and returns a
``CheckResult`` whose ``detail`` says what it saw, so a failure explains itself.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str
