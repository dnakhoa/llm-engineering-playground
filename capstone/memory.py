"""Conversation memory — Module 11 applied."""
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Turn:
    role: str   # "user" or "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ConversationMemory:
    """Short-term sliding window memory."""

    def __init__(self, max_turns: int = 10, max_chars: int = 4000):
        self.max_turns = max_turns
        self.max_chars = max_chars
        self._turns: deque[Turn] = deque(maxlen=max_turns * 2)

    def add(self, role: str, content: str):
        self._turns.append(Turn(role=role, content=content))

    def to_string(self) -> str:
        """Return conversation history as a single string, trimmed to max_chars."""
        lines = [f"{t.role.capitalize()}: {t.content}" for t in self._turns]
        history = "\n".join(lines)
        if len(history) > self.max_chars:
            history = "...[truncated]\n" + history[-self.max_chars:]
        return history

    def clear(self):
        self._turns.clear()

    def __len__(self):
        return len(self._turns)
