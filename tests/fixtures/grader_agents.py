"""Reader agents for testing the Grader, loaded by importable reference.

None of them is the reference Flagship Agent, so the Grader treats each as the
Reader's own work. They call Actions directly and never ask a model, so an
Offline run grades them without a recording of their requests.
"""
from __future__ import annotations

from llm.types import ToolCall


def upgrades_without_a_model(customer_turn, env):
    """Resolves the Spine 1 Case: one change_plan to Pro on the customer's account."""
    env.act(ToolCall(id="c1", name="change_plan",
                     arguments={"account_id": "acct_1001", "plan": "pro"}))
    return "You are on Pro."


def raises(customer_turn, env):
    """Breaks the agent contract by raising instead of replying."""
    raise RuntimeError("forgot to handle the customer turn")
