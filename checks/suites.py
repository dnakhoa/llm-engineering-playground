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
from typing import Callable, Tuple

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


def discover_suites() -> Tuple[Suite, ...]:
    """Every ``checks.spine_<n>_<name>`` suite, in module order."""
    package = Path(__file__).resolve().parent
    suites = []
    for info in pkgutil.iter_modules([str(package)]):
        if not _SUITE_NAME.match(info.name):
            continue
        module = importlib.import_module("checks." + info.name)
        suites.append(
            Suite(
                module=int(module.MODULE),
                title=str(module.TITLE),
                cases=tuple(module.CASES),
                checks=tuple(module.CHECKS),
            )
        )
    return tuple(sorted(suites, key=lambda suite: suite.module))
