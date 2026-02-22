import time
from typing import Optional
from loguru import logger

class Colors:
    GREY = '\x1b[38;21m'
    BLUE = '\x1b[38;5;39m'
    YELLOW = '\x1b[33m'
    RED = '\x1b[31m'
    GREEN = '\x1b[32m'
    MAGENTA = '\x1b[35m'
    CYAN = '\x1b[36m'
    RESET = '\x1b[0m'

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
    def __init__(self, state: 'SessionState', name: str):
        self.state = state
        self.name = name
        self.start: Optional[float] = None
    
    async def __aenter__(self):
        self.start = time.time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.start is not None:
            elapsed = time.time() - self.start
            self.state.timers[self.name] = elapsed
            color_print('debug', f"[Timer] {self.name}: {elapsed:.2f}s")

def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins}m {secs:.1f}s"
