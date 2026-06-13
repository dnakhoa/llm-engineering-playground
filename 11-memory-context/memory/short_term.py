"""Short-term conversation memory — buffer and sliding window."""

from collections import deque
from typing import List, Optional


class ConversationBuffer:
    """Maintains recent conversation history with token-aware trimming."""

    def __init__(self, max_messages: int = 20, max_tokens: int = 4000):
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.buffer: deque[dict] = deque(maxlen=max_messages * 2)

    def add_message(self, role: str, content: str):
        self.buffer.append({"role": role, "content": content})
        while self._estimate_tokens() > self.max_tokens and len(self.buffer) > 2:
            self.buffer.popleft()
            self.buffer.popleft()

    def get_messages(self) -> List[dict]:
        return list(self.buffer)

    def get_summary_context(self) -> str:
        msgs = list(self.buffer)
        if len(msgs) < 4:
            return "\n".join(m["content"] for m in msgs)
        return "\n".join(m["content"] for m in msgs[:2] + msgs[-(self.max_messages - 2):])

    def clear(self):
        self.buffer.clear()

    def _estimate_tokens(self) -> int:
        return sum(len(m["content"]) // 4 for m in self.buffer)

    def __len__(self):
        return len(self.buffer)


class SlidingWindowMemory(ConversationBuffer):
    """Sliding window with configurable overlap."""

    def __init__(self, window_size: int = 10, overlap: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.overlap = overlap

    def get_window(self) -> List[dict]:
        messages = list(self.buffer)
        if len(messages) <= self.window_size:
            return messages
        start = max(0, len(messages) - self.window_size - self.overlap)
        return messages[start:]
