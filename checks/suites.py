"""The Check suites, one per Spine module, found by name.

A suite is a module ``checks/spine_<n>_<name>.py`` that defines ``MODULE``,
``TITLE``, ``CASES`` (Case ids, or paths to Case files) and ``CHECKS``. Adding
Spine module N's suite is adding that file; nothing here changes.
"""
from __future__ import annotations

import importlib
import pkgutil
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from company.runner import Outcome

from . import CheckResult

Check = Callable[[Outcome], CheckResult]

_SUITE_NAME = re.compile(r"^spine_(\d+)_\w+$")


@dataclass(frozen=True)
class Suite:
    """One Spine module's Checks and the Cases they run on."""

    module: int
    title: str
    cases: Tuple[str, ...]
    checks: Tuple[Check, ...]


def _suite_modules(before: Optional[int] = None) -> List[Any]:
    """The ``checks.spine_<n>_<name>`` modules, imported, for every n, or for
    every n below ``before`` (so a suite can ask for the ones before it
    without importing itself)."""
    package = Path(__file__).resolve().parent
    found = []
    for info in pkgutil.iter_modules([str(package)]):
        match = _SUITE_NAME.match(info.name)
        if not match or (before is not None and int(match.group(1)) >= before):
            continue
        found.append(importlib.import_module("checks." + info.name))
    return sorted(found, key=lambda module: int(module.MODULE))


def cases_before(module: int) -> Tuple[str, ...]:
    """Every Case the suites of modules 1 to ``module - 1`` run, once each, in
    module order.

    For a suite whose Checks hold on every Case so far, such as Spine 4's
    trace Checks: its Cases are derived here, never copied (ADR 0006), and
    ``on_every_case`` carries its Checks to every later suite's Cases.
    """
    cases: Dict[str, None] = {}
    for suite in _suite_modules(before=module):
        cases.update(dict.fromkeys(suite.CASES))
    return tuple(cases)


def discover_suites() -> Tuple[Suite, ...]:
    """Every ``checks.spine_<n>_<name>`` suite, in module order."""
    suites = []
    for module in _suite_modules():
        suites.append(
            Suite(
                module=int(module.MODULE),
                title=str(module.TITLE),
                cases=tuple(module.CASES),
                checks=tuple(module.CHECKS),
            )
        )
    return tuple(sorted(suites, key=lambda suite: suite.module))
