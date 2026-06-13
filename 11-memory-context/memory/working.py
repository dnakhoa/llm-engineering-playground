"""Working memory — context assembly for LLM calls."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import json


@dataclass
class MemoryContext:
    """Assembled context for an LLM call."""
    system_prompt: str = ""
    conversation_history: List[dict] = field(default_factory=list)
    retrieved_memories: List[str] = field(default_factory=list)
    task_context: Dict = field(default_factory=dict)
    tool_results: List[str] = field(default_factory=list)

    def to_messages(self) -> List[dict]:
        messages = []
        if self.system_prompt or self.retrieved_memories:
            parts = [self.system_prompt]
            if self.retrieved_memories:
                parts.append("\n\nRelevant memories:\n" + "\n".join(f"- {m}" for m in self.retrieved_memories))
            messages.append({"role": "system", "content": "\n".join(parts)})
        messages.extend(self.conversation_history)
        if self.task_context:
            messages.append({"role": "user", "content": f"Task context: {json.dumps(self.task_context)}"})
        if self.tool_results:
            messages.append({"role": "assistant", "content": "Tool results:\n" + "\n".join(self.tool_results)})
        return messages

    def count_tokens(self, encoder=None) -> int:
        if encoder is None:
            return sum(len(m.get("content", "")) // 4 for m in self.to_messages())
        return sum(len(encoder.encode(m.get("content", ""))) for m in self.to_messages())


class ContextAssembler:
    """Assembles context within token limits using priority ordering."""

    def __init__(self, max_context_tokens: int = 8000, priority_order: List[str] = None):
        self.max_tokens = max_context_tokens
        self.priority_order = priority_order or [
            "system_prompt", "recent_conversation", "task_context",
            "relevant_memories", "tool_results", "older_conversation",
        ]

    def assemble(
        self,
        system_prompt: str,
        conversation: List[dict],
        memories: List[str],
        task_context: Dict,
        tool_results: List[str],
        encoder=None,
    ) -> MemoryContext:
        context = MemoryContext(system_prompt=system_prompt, task_context=task_context)
        context.conversation_history = conversation[-10:]
        current_tokens = context.count_tokens(encoder)

        if current_tokens < self.max_tokens * 0.7 and memories:
            context.retrieved_memories = memories[:5]
            current_tokens = context.count_tokens(encoder)

        if tool_results and current_tokens < self.max_tokens * 0.9:
            context.tool_results = tool_results[:3]

        return context
