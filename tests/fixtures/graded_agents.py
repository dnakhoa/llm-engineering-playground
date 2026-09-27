"""A seeded regression in the reference agent, for the Graded suite to catch.

``refunds_twice`` is the Spine 2 reference agent (``flagship.knowledge:run``)
with one bug of the kind a retry wrapper introduces: every ``issue_refund`` it
runs is sent to the Backend twice. The model only ever sees the first result,
so every request is one the recordings hold and the whole run replays Offline.

On the prorated-refund Case the first refund uses up what the invoice owes,
so the Backend refuses the second and the final state is exactly right. The
Knowledge suite, which reads only Backend state, passes it. The Graded suite
has to catch it from the attempts.

``upgrades_then_says_nothing`` and ``upgrades_then_hits_the_step_limit`` are
the Spine 4 reference agent (``flagship.observed:run``) except on
``upgrade-to-pro``: it upgrades the account as asked, then answers the customer
with an empty reply, or hits the step limit. The plan ends on Pro after one
change_plan, so Spine 1's Check passes, and the Knowledge suite does not run
that Case. The Outcome is not resolved, so the Graded suite has to fail it.

Loaded by importable reference (``tests.fixtures.graded_agents:refunds_twice``),
the way a Reader's agent is.
"""
from __future__ import annotations

from company.runner import StepLimitReached
from flagship.knowledge import run as reference_agent
from flagship.observed import run as observed_agent

#: Only the upgrade-to-pro Case's customer says this.
_UPGRADE_TO_PRO = "Please upgrade us to Pro."


class _SendsRefundsTwice:
    """The Case's environment, except that ``act`` sends each refund twice."""

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def act(self, call):
        result = self._env.act(call)
        if call.name == "issue_refund":
            self._env.act(call)  # the bug: a retry that never checked the first result
        return result


def refunds_twice(customer_turn, env):
    """The reference agent, with every refund sent to the Backend twice."""
    return reference_agent(customer_turn, _SendsRefundsTwice(env))


def upgrades_then_says_nothing(customer_turn, env):
    """The Spine 4 reference agent, with an empty reply once it has upgraded to Pro."""
    reply = observed_agent(customer_turn, env)
    return "" if _UPGRADE_TO_PRO in customer_turn else reply


def upgrades_then_hits_the_step_limit(customer_turn, env):
    """The Spine 4 reference agent, stopped by the step limit once it has upgraded to Pro."""
    reply = observed_agent(customer_turn, env)
    if _UPGRADE_TO_PRO in customer_turn:
        raise StepLimitReached(env.step_limit)
    return reply
