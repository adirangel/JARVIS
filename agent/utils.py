import time
from typing import Optional
from loguru import logger


def color_print(level: str, message: str):
    """Single-channel log via loguru. No more duplicate print() + logger calls."""
    _dispatch = {
        'thought': logger.debug,
        'info':    logger.info,
        'warn':    logger.warning,
        'error':   logger.error,
        'success': logger.success,
        'debug':   logger.debug,
    }
    _dispatch.get(level, logger.info)(message)


class DebugTimer:
    def __init__(self, state, name: str):
        self.state = state
        self.name = name
        self.start: Optional[float] = None

    async def __aenter__(self):
        self.start = time.time()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.start is not None:
            elapsed = time.time() - self.start
            if hasattr(self.state, 'timers') and isinstance(self.state.timers, dict):
                self.state.timers[self.name] = elapsed
            logger.debug(f"[Timer] {self.name}: {elapsed:.2f}s")

def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins}m {secs:.1f}s"
