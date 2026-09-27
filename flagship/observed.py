"""Spine 4: the Spine 2 agent, traced, and saying who it is.

Most of the tracing needs no code here. The Case runner opens an
``invoke_agent`` span for the Case, and the environment turns every
``env.complete`` into a ``chat`` span and every ``env.act`` into an
``execute_tool`` span, with tokens and cost on them. That is the same place an
instrumentation library would hook into a framework, and it is why the loop
must send every tool call through ``env.act``: a tool the agent runs some other
way leaves no span, and the Observed Checks fail it.

What the environment cannot know is which agent this is. The conventions put
that on the agent span as ``gen_ai.agent.name`` and ``gen_ai.agent.version``,
"if provided by the application", so this agent provides them. In a tracing
backend that is the filter that finds this agent's traces among every other
service's, and the version that tells a Tuesday regression from a Monday one.

The loop is Spine 2's, untouched, so every request is the one recorded: the
name goes on a span, never into a prompt.
"""
from __future__ import annotations

from .knowledge import run as knowledge_run

AGENT_NAME = "acme-support"
AGENT_VERSION = "spine-4"
AGENT_DESCRIPTION = "Acme Notes' support agent: answers from the Knowledge Base, acts on the Backend."


def run(customer_turn, env):
    env.describe_agent(AGENT_NAME, version=AGENT_VERSION, description=AGENT_DESCRIPTION)
    return knowledge_run(customer_turn, env)
