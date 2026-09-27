"""The provider layer: one call, many providers, no 400s.

Current models disagree about basic parameters. Claude models from Opus 4.7 on
reject a non-default ``temperature``; current OpenAI models need the Responses
API for tool calls; Gemini calls its effort control ``thinkingLevel``; DeepSeek
accepts a temperature in thinking mode and then ignores it. So this layer does
not pass one set of arguments through. It reads the model registry, drops or
translates whatever the chosen model cannot act on, and says so in
``Response.adjustments`` instead of swallowing it.

    from llm import CallOptions, Message, complete
    from llm.transport import HttpTransport

    answer = complete(
        model="claude-sonnet-5",
        messages=[Message.user("What is Acme's refund window?")],
        options=CallOptions(effort="high", temperature=0.7),
        transport=HttpTransport(provider="anthropic"),
    )
    print(answer.text)
    for note in answer.adjustments:
        print(note)      # dropped temperature=0.7: claude-sonnet-5 does not ...

See ADR 0002 (docs/adr/0002-provider-agnostic-capability-aware.md).
"""
from __future__ import annotations

from .call import build_request, complete  # noqa: F401
from .capabilities import RequestPlan, plan_request  # noqa: F401
from .registry import (  # noqa: F401
    EFFORT_LADDER,
    REGISTRY_PATH,
    ApiSurface,
    ModelSpec,
    Registry,
    load_registry,
)
from .replay import (  # noqa: F401
    RecordingTransport,
    ReplayMismatchError,
    ReplayTransport,
)
from .types import (  # noqa: F401
    Adjustment,
    CallOptions,
    Message,
    ProviderRequest,
    Response,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
)

__all__ = [
    "Adjustment",
    "ApiSurface",
    "CallOptions",
    "EFFORT_LADDER",
    "Message",
    "ModelSpec",
    "ProviderRequest",
    "REGISTRY_PATH",
    "RecordingTransport",
    "Registry",
    "ReplayMismatchError",
    "ReplayTransport",
    "RequestPlan",
    "Response",
    "ToolCall",
    "ToolResult",
    "ToolSpec",
    "Usage",
    "build_request",
    "complete",
    "load_registry",
    "plan_request",
]
