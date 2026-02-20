"""Heartbeat: Every 30 minutes, check memory for pending tasks/reminders.

Execute if possible, speak witty summary via TTS.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Optional


def load_config() -> dict:
    import yaml
    base = Path(__file__).parent
    cfg_path = base / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def heartbeat_job(
    memory: Any,
    tts_speak: Callable[[str], None],
    llm_invoke: Optional[Callable[[str], str]] = None,
) -> None:
    """Run heartbeat: check tasks/reminders, execute, speak summary."""
    pending = getattr(memory, "get_pending_tasks", lambda: [])()
    reminders = getattr(memory, "get_reminders", lambda: [])()

    if not pending and not reminders:
        return

    # Mark reminders as done (user will be notified)
    for r in reminders:
        getattr(memory, "mark_reminder_done", lambda x: None)(r.get("id"))

    # Build summary
    parts = []
    if pending:
        parts.append(f"{len(pending)} pending task(s)")
    if reminders:
        parts.append(f"{len(reminders)} reminder(s)")

    summary = "; ".join(parts)
    if llm_invoke:
        try:
            prompt = f"Sir has: {summary}. One brief, witty sentence to speak aloud. JARVIS character."
            text = llm_invoke(prompt)
            if text:
                tts_speak(text)
        except Exception:
            tts_speak(f"Sir, you have {summary}.")
    else:
        tts_speak(f"Sir, you have {summary}.")


def start_heartbeat(
    memory: Any,
    tts_speak: Callable[[str], None],
    llm_invoke: Optional[Callable[[str], str]] = None,
    interval_minutes: int = 30,
) -> Any:
    """Start APScheduler heartbeat job."""
    from apscheduler.schedulers.background import BackgroundScheduler

    scheduler = BackgroundScheduler()
    scheduler.add_job(
        heartbeat_job,
        "interval",
        minutes=interval_minutes,
        args=[memory, tts_speak],
        kwargs={"llm_invoke": llm_invoke},
    )
    scheduler.start()
    return scheduler
