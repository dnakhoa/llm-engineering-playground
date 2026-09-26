"""Spine 1: the agent loop. Ask the model, run the tools it asks for, repeat.

The whole agent is the ``run`` function below. It follows the agent contract
the Case runner expects: it gets the customer's turn and an environment, and
returns its final reply. The environment gives it the Actions as tools
(``env.tools``, ``env.act``) and the model (``env.complete``, which is
``llm.complete`` with the model and transport already chosen).

Two things end the loop:

* **The model is done.** A reply with no tool calls is the model's answer.
* **The step limit.** ``env.step_limit`` model calls, then stop. A model that
  keeps calling tools would otherwise spend money for ever.
"""
from __future__ import annotations

from company.runner import StepLimitReached
from llm.types import Message

SYSTEM = (
    "You are the support agent for Acme Notes, a note-taking app with Free, Pro "
    "and Team plans. Resolve the customer's request using your tools. Look the "
    "account up before you change it. Only act on the customer's own account. "
    "When the request is done, or cannot be done, reply to the customer briefly."
)


def run(customer_turn, env):
    messages = [Message.user(customer_turn)]

    for _ in range(env.step_limit):
        response = env.complete(messages, tools=env.tools, system=SYSTEM)
        messages.append(Message.assistant(response.text or None, response.tool_calls))

        if not response.tool_calls:  # stop condition: the model has answered
            return response.text

        results = [env.act(call) for call in response.tool_calls]
        messages.append(Message.tool(results))

    # Stop condition: out of steps. The Case runner reports it on the Outcome.
    raise StepLimitReached(env.step_limit)
