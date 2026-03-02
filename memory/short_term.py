"""Short-term (in-session) conversation memory. Sliding window of recent turns."""

from __future__ import annotations

from typing import Any, Dict, List


class ShortTermMemory:
    """In-memory sliding window of recent user/assistant turns for LLM context."""

    def __init__(self, max_turns: int = 15):
        self._max_turns = max_turns
        self._turns: List[Dict[str, str]] = []

    def add(self, role: str, content: str) -> None:
        """Append a message (user or assistant) to recent history."""
        self._turns.append({"role": role, "content": content})
        # Keep only last max_turns pairs (each turn = user + assistant)
        while len(self._turns) > self._max_turns * 2:
            self._turns.pop(0)

    def get_context(self) -> List[Dict[str, str]]:
        """Return recent messages as list of {role, content} for LLM."""
        return list(self._turns)
