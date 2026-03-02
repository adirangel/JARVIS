"""Uptime endpoint - session duration, command count."""

import time
from fastapi import APIRouter

router = APIRouter(prefix="/api", tags=["uptime"])

# App state - initialized on first request
_start_time = None
_command_count = 0


def _ensure_started():
    global _start_time
    if _start_time is None:
        _start_time = time.time()


def _get_system_load() -> float:
    """Get system load (CPU percent as proxy)."""
    try:
        import psutil
        return psutil.cpu_percent(interval=0.1)
    except Exception:
        return 0.0


@router.get("/uptime")
async def get_uptime():
    """Return session uptime and command count."""
    _ensure_started()
    elapsed = time.time() - _start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return {
        "uptime_seconds": round(elapsed, 1),
        "uptime_formatted": formatted,
        "session": 1,
        "commands": _command_count,
        "system_load": round(_get_system_load(), 1),
    }


def increment_command_count():
    """Call when a conversation message is processed."""
    global _command_count
    _command_count += 1
