"""Spine 2: the agent loop, with memory across a Case's turns.

The loop is Spine 1's (``flagship.loop``) with one change: the transcript
outlives the turn. Spine 1 started every call to ``run`` from an empty list,
so a customer's second message arrived with nothing before it. Here the
transcript lives in ``env.memory``, which the Case runner keeps for the whole
Case and empties before the next, so "please go ahead" on turn two still
knows which account turn one was about.

Knowledge needs no new code in the loop. A Case that declares the Knowledge
Base puts ``search_knowledge_base`` in ``env.tools``, next to the Actions it
declares, and ``env.act`` answers it from the retriever. The model decides when
to look a policy up, the same way it decides when to look an account up.

The system prompt is Spine 1's, word for word, on purpose: it is part of every
recorded request, so changing it would mean re-recording every Case.
"""
from __future__ import annotations

from company.runner import StepLimitReached
from llm.types import Message

from .loop import SYSTEM


def run(customer_turn, env):
    messages = env.memory.setdefault("transcript", [])  # memory: kept across turns
    messages.append(Message.user(customer_turn))

    for _ in range(env.step_limit):
        response = env.complete(messages, tools=env.tools, system=SYSTEM)
        messages.append(Message.assistant(response.text or None, response.tool_calls))

        if not response.tool_calls:  # stop condition: the model has answered
            return response.text

        results = [env.act(call) for call in response.tool_calls]
        messages.append(Message.tool(results))

    # Stop condition: out of steps. The Case runner reports it on the Outcome.
    raise StepLimitReached(env.step_limit)
