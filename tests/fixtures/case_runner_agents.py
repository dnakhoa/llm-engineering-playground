"""Agents a Reader might write, used to test the Case runner through its seam.

Each one follows the agent contract: a callable taking the Case's customer turn
and an environment, returning the final reply. They are loaded by importable
reference (``tests.fixtures.case_runner_agents:<name>``), the same way the
runner loads a Reader's own agent.
"""
from __future__ import annotations

from llm.types import Message, ToolCall


def chats_but_never_acts(customer_turn, env):
    """Answers politely and changes nothing."""
    response = env.complete([Message.user(customer_turn)])
    return response.text


def loops_for_ever(customer_turn, env):
    """Keeps asking the model, with no step limit of its own."""
    messages = [Message.user(customer_turn)]
    while True:
        response = env.complete(messages, tools=env.tools)
        messages.append(Message.assistant(response.text, response.tool_calls))


def changes_someone_elses_plan(customer_turn, env):
    """Tries to move another customer's account, however it was talked into it."""
    env.act(
        ToolCall(
            id="call_x",
            name="change_plan",
            arguments={"account_id": "acct_1002", "plan": "free"},
        )
    )
    return "Done."


def claims_without_acting(customer_turn, env):
    """Asks the model, passes on its "Done!", and never calls change_plan."""
    response = env.complete([Message.user(customer_turn)], tools=env.tools)
    return response.text


def upgrades_downgrades_and_upgrades_again(customer_turn, env):
    """Ends on Pro, but only after three plan changes."""
    for index, plan in enumerate(("pro", "free", "pro")):
        env.act(
            ToolCall(
                id="call_{}".format(index),
                name="change_plan",
                arguments={"account_id": "acct_1001", "plan": plan},
            )
        )
    return "You're on Pro now."


def upgrades_someone_elses_account(customer_turn, env):
    """Upgrades the wrong account: acct_1003 instead of the Case's acct_1001."""
    env.act(
        ToolCall(
            id="call_y",
            name="change_plan",
            arguments={"account_id": "acct_1003", "plan": "pro"},
        )
    )
    return "Done! You're on Pro now."


def refunds_on_a_case_that_offers_no_refunds(customer_turn, env):
    """Calls issue_refund, which the upgrade-to-pro Case never offered."""
    env.act(
        ToolCall(
            id="call_r",
            name="issue_refund",
            arguments={"account_id": "acct_1001", "invoice_id": "inv_2002", "amount_usd": 1},
        )
    )
    return "Refunded."


def loop_with_another_system_prompt(customer_turn, env):
    """The reference loop's first call, with a reworded system prompt."""
    response = env.complete(
        [Message.user(customer_turn)],
        tools=env.tools,
        system="You are Acme Notes' friendly support agent.",
    )
    return response.text
