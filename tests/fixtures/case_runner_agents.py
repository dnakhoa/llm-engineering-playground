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
