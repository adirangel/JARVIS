"""Voice loop session state machine helpers.

Wake-only mode waits for wake word.
Active-session mode allows follow-ups without wake word until timeout or explicit end command.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable


class VoiceLoopState(str, Enum):
    WAKE_ONLY = "wake_only"
    ACTIVE_SESSION = "active_session"


DEFAULT_END_COMMANDS = (
    "goodbye",
    "good bye",
    "stop",
    "end session",
    "that's all",
    "cancel",
)


_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _tokenize(text: str) -> set[str]:
    return {m.group(0) for m in _WORD_RE.finditer(text)}


def contains_end_command(text: str, commands: Iterable[str] = DEFAULT_END_COMMANDS) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    words = _tokenize(normalized)
    for cmd in commands:
        c = _normalize_text(cmd)
        if not c:
            continue
        if " " in c:
            if c in normalized:
                return True
            continue
        if c in words:
            return True
    return False


@dataclass
class VoiceSession:
    silence_timeout: float = 15.0
    end_commands: tuple[str, ...] = field(default_factory=lambda: tuple(DEFAULT_END_COMMANDS))
    _state: VoiceLoopState = field(default=VoiceLoopState.WAKE_ONLY, init=False)
    _last_activity: float = field(default=0.0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def activate(self) -> None:
        with self._lock:
            self._state = VoiceLoopState.ACTIVE_SESSION
            self._last_activity = time.monotonic()

    def end(self) -> None:
        with self._lock:
            self._state = VoiceLoopState.WAKE_ONLY
            self._last_activity = 0.0

    def touch(self) -> None:
        with self._lock:
            if self._state == VoiceLoopState.ACTIVE_SESSION:
                self._last_activity = time.monotonic()

    def is_active(self) -> bool:
        with self._lock:
            return self._state == VoiceLoopState.ACTIVE_SESSION

    def state(self) -> VoiceLoopState:
        with self._lock:
            return self._state

    def time_remaining(self) -> float:
        with self._lock:
            if self._state != VoiceLoopState.ACTIVE_SESSION:
                return 0.0
            if self.silence_timeout <= 0:
                return float("inf")
            elapsed = time.monotonic() - self._last_activity
            return max(0.0, self.silence_timeout - elapsed)

    def timed_out(self) -> bool:
        rem = self.time_remaining()
        return rem == 0.0 if self.silence_timeout > 0 else False

    def should_end_for_text(self, text: str) -> bool:
        return contains_end_command(text, self.end_commands)

