"""Hierarchical memory — L1 (working) + L2 (short-term) + L3 (long-term)."""

from typing import Dict, Optional
from .short_term import ConversationBuffer
from .long_term import VectorMemory
from .working import ContextAssembler, MemoryContext


class HierarchicalMemory:
    """
    Multi-level memory system:
    - L1: Working memory (immediate context)
    - L2: Short-term buffer (current session)
    - L3: Long-term storage (persistent)
    """

    def __init__(
        self,
        user_id: str,
        session_id: str = "default",
        l1_max_tokens: int = 4000,
        l2_max_messages: int = 50,
        l3_persist_dir: str = "./memory_db",
    ):
        self.user_id = user_id
        self.session_id = session_id
        self.short_term = ConversationBuffer(max_messages=l2_max_messages, max_tokens=8000)
        self.long_term = VectorMemory(persist_directory=f"{l3_persist_dir}/{user_id}")
        self.assembler = ContextAssembler(max_context_tokens=l1_max_tokens)

    def add_interaction(self, user_input: str, assistant_response: str, extract_memories: bool = True):
        self.short_term.add_message("user", user_input)
        self.short_term.add_message("assistant", assistant_response)
        if extract_memories:
            self._extract_and_store(user_input, assistant_response)

    def _extract_and_store(self, user_input: str, assistant_response: str):
        preference_kw = ["i prefer", "i like", "i hate", "always", "never"]
        if any(kw in user_input.lower() for kw in preference_kw):
            self.long_term.store_memory(user_input, {"user_id": self.user_id}, "preference")

        fact_kw = ["i am", "i work", "i live", "my name is"]
        if any(kw in user_input.lower() for kw in fact_kw):
            self.long_term.store_memory(user_input, {"user_id": self.user_id}, "fact")

    def get_context(self, current_query: str, system_prompt: str = "", task_context: Dict = None) -> MemoryContext:
        memories = self.long_term.retrieve_memories(current_query, k=5)
        return self.assembler.assemble(
            system_prompt=system_prompt,
            conversation=self.short_term.get_messages(),
            memories=[m["content"] for m in memories],
            task_context=task_context or {},
            tool_results=[],
        )

    def end_session(self, archive: bool = True):
        if archive:
            summary = f"Session {self.session_id}: {len(self.short_term)} messages."
            self.long_term.store_memory(summary, {"user_id": self.user_id, "session_id": self.session_id}, "session")
        self.short_term.clear()
