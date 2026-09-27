"""Agents that get the prorated refund wrong and every other Case right.

Each one is the Spine 2 reference agent (``flagship.knowledge:run``) on every
Case but ``downgrade-with-prorated-refund``, where it does its own wrong thing
without asking the model. So when the Checks CLI grades one, the only Case in
modules 1 and 2 that can fail is the prorated refund: if the Knowledge suite
still reports "Passed through module 2 of 2", its Checks let a wrong refund
through.

They are loaded by importable reference
(``tests.fixtures.knowledge_agents:<name>``), the way a Reader's agent is.
"""
from __future__ import annotations

from flagship.knowledge import run as reference_agent
from llm.types import ToolCall

#: Only the prorated-refund Case's customer names this account.
PRORATED_ACCOUNT = "acct_1002"
PRORATED_INVOICE = "inv_2002"


def _on_the_prorated_case(wrong):
    def agent(customer_turn, env):
        if PRORATED_ACCOUNT not in customer_turn:
            return reference_agent(customer_turn, env)
        return wrong(env)

    agent.__name__ = wrong.__name__
    agent.__doc__ = wrong.__doc__
    return agent


def _downgrade(env):
    env.act(ToolCall("w1", "change_plan", {"account_id": PRORATED_ACCOUNT, "plan": "free"}))


def _refund(env, call_id, amount_usd):
    env.act(ToolCall(call_id, "issue_refund", {
        "account_id": PRORATED_ACCOUNT, "invoice_id": PRORATED_INVOICE,
        "amount_usd": amount_usd}))


@_on_the_prorated_case
def does_nothing(env):
    """Changes nothing: no downgrade, no refund."""
    return "Sorry, can't help."


@_on_the_prorated_case
def refunds_five_dollars(env):
    """Downgrades and refunds $5.00: allowed by the Action, not what the article gives."""
    _downgrade(env)
    _refund(env, "w2", 5.00)
    return "Done."


@_on_the_prorated_case
def refunds_the_whole_invoice(env):
    """Downgrades and asks for all $12.00 back, which the Action refuses."""
    _downgrade(env)
    _refund(env, "w2", 12.00)
    return "Done."


@_on_the_prorated_case
def refunds_the_policy_amount_in_two_parts(env):
    """Downgrades and refunds $9.00 then $0.20: the right total, in two refunds."""
    _downgrade(env)
    _refund(env, "w2", 9.00)
    _refund(env, "w3", 0.20)
    return "Done."


@_on_the_prorated_case
def downgrades_without_refunding(env):
    """Downgrades and refunds nothing."""
    _downgrade(env)
    return "You're on Free now."


WRONG_ON_THE_PRORATED_CASE = (
    "does_nothing",
    "refunds_five_dollars",
    "refunds_the_whole_invoice",
    "refunds_the_policy_amount_in_two_parts",
    "downgrades_without_refunding",
)
