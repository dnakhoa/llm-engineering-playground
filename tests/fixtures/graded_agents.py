"""A seeded regression in the reference agent, for the Graded suite to catch.

``refunds_twice`` is the Spine 2 reference agent (``flagship.knowledge:run``)
with one bug of the kind a retry wrapper introduces: every ``issue_refund`` it
runs is sent to the Backend twice. The model only ever sees the first result,
so every request is one the recordings hold and the whole run replays Offline.

On the prorated-refund Case the first refund uses up what the invoice owes,
so the Backend refuses the second and the final state is exactly right. The
Knowledge suite, which reads only Backend state, passes it. The Graded suite
has to catch it from the attempts.

Loaded by importable reference (``tests.fixtures.graded_agents:refunds_twice``),
the way a Reader's agent is.
"""
from __future__ import annotations

from flagship.knowledge import run as reference_agent


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
