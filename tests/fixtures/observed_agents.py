"""An agent whose Backend work is right and whose trace is not.

``searches_without_a_span`` is the Spine 4 reference agent, except that it
answers Knowledge Base searches itself, with the same retriever and the same
result, instead of through ``env.act``. Every request matches the recording
and every Graded Check passes; only the trace shows the search never happened
as far as a tracing backend can tell.

Loaded by importable reference
(``tests.fixtures.observed_agents:searches_without_a_span``), the way a
Reader's agent is.
"""
from __future__ import annotations

from company.knowledge import SEARCH_TOOL, run_search
from company.runner import StepLimitReached
from flagship.loop import SYSTEM
from flagship.observed import AGENT_NAME
from llm.types import Message


def searches_without_a_span(customer_turn, env):
    env.describe_agent(AGENT_NAME)
    messages = env.memory.setdefault("transcript", [])
    messages.append(Message.user(customer_turn))

    for _ in range(env.step_limit):
        response = env.complete(messages, tools=env.tools, system=SYSTEM)
        messages.append(Message.assistant(response.text or None, response.tool_calls))
        if not response.tool_calls:
            return response.text
        results = [
            run_search(env.knowledge, call) if call.name == SEARCH_TOOL.name else env.act(call)
            for call in response.tool_calls
        ]
        messages.append(Message.tool(results))

    raise StepLimitReached(env.step_limit)
