"""Spine 1 (Loop) Checks.

They assert on Backend state and on the Actions that actually ran, never on
what the agent said. "I've upgraded you!" with no plan change fails.
"""
from __future__ import annotations

from company.runner import Outcome

from . import CheckResult


def plan_changed_to_pro_exactly_once(outcome: Outcome) -> CheckResult:
    """The customer's account ends on Pro, moved there by exactly one Action."""
    name = "plan changed to Pro exactly once"
    account = outcome.case.customer_account_id
    upgrades = [
        record
        for record in outcome.actions_executed
        if record.name == "change_plan"
        and record.arguments.get("account_id") == account
        and record.arguments.get("plan") == "pro"
    ]
    plan = outcome.final_state["accounts"][account]["plan"]

    if plan == "pro" and len(upgrades) == 1:
        return CheckResult(name, True, "{} is on Pro after one change_plan.".format(account))
    if outcome.step_limit_reached:
        why = "the agent hit the step limit ({} model calls)".format(outcome.step_limit)
    elif not upgrades:
        why = "no change_plan to Pro ran for {}".format(account)
    else:
        why = "{} change_plan calls to Pro ran".format(len(upgrades))
    return CheckResult(
        name, False, "{} ends on {}: {}.".format(account, plan, why)
    )


CHECKS = (plan_changed_to_pro_exactly_once,)
