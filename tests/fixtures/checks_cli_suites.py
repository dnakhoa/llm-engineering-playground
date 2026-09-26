"""Stand-in Check suites for testing the Checks CLI before Spine 2 exists.

Module 2 here replays the same recording as Spine 1 under another Case id, so
a run through module 2 costs exactly twice a run through module 1. That makes
the Spend Cap arithmetic in the tests something you can do by hand.
"""
from __future__ import annotations

from pathlib import Path

from checks import CheckResult, live_only
from checks.spine_1_loop import plan_changed_to_pro_exactly_once
from checks.suites import Suite

AGAIN = str(Path(__file__).parent / "checks_cli" / "upgrade-to-pro-again.json")


@live_only
def reply_is_polite(outcome):
    """Stands in for an LLM-as-judge rubric: it needs a live model to grade."""
    return CheckResult("reply is polite (judge)", True, "judged polite.")


LOOP = Suite(
    module=1,
    title="Loop",
    cases=("upgrade-to-pro",),
    checks=(plan_changed_to_pro_exactly_once,),
)

SECOND = Suite(
    module=2,
    title="Second",
    cases=(AGAIN,),
    checks=(plan_changed_to_pro_exactly_once, reply_is_polite),
)

SUITES = (LOOP, SECOND)
