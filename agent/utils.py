import time
from typing import Optional
from loguru import logger

class Colors:
    GREY = '[38;21m'
    BLUE = '[38;5;39m'
    YELLOW = '[33m'
    RED = '[31m'
    GREEN = '[32m'
    MAGENTA = '[35m'
    CYAN = '[36m'
    RESET = '[0m'

def color_print(level: str, message: str):
    prefix = {
        'thought': Colors.MAGENTA,
        'info': Colors.BLUE,
        'warn': Colors.YELLOW,
        'error': Colors.RED,
        'success': Colors.GREEN,
        'debug': Colors.CYAN
    }.get(level, Colors.GREY)
    formatted = f"[{level.upper()}] {message}"
    print(prefix + formatted + Colors.RESET)
    {
        'thought': logger.debug,
        'info': logger.info,
        'warn': logger.warning,
        'error': logger.error,
        'success': logger.success,
        'debug': logger.debug
    }.get(level, logger.info)(message)

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
            # Only store timers if state has a timers dict
            if hasattr(self.state, 'timers') and isinstance(self.state.timers, dict):
                self.state.timers[self.name] = elapsed
            color_print('debug', f"[Timer] {self.name}: {elapsed:.2f}s")

def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins}m {secs:.1f}s"
