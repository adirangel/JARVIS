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
    """Run heartbeat: check tasks/reminders, speak witty summary every 30 min (even when idle)."""
    pending = getattr(memory, "get_pending_tasks", lambda: [])()
    reminders = getattr(memory, "get_reminders", lambda: [])()

    # Mark reminders as done (user will be notified) - fix: pass each id explicitly
    mark_done = getattr(memory, "mark_reminder_done", None)
    if mark_done and callable(mark_done):
        for r in reminders:
            rid = r.get("id")
            if rid is not None:
                mark_done(rid)

    # Build summary - always speak (even when idle)
    parts = []
    if pending:
        parts.append(f"{len(pending)} pending task(s)")
    if reminders:
        parts.append(f"{len(reminders)} reminder(s)")
    if not parts:
        summary = "idle, no pending tasks or reminders"
    else:
        summary = "; ".join(parts)

    if llm_invoke:
        try:
            prompt = (
                f"JARVIS heartbeat. Sir has: {summary}. "
                "One brief, witty sentence to speak aloud. Dry British wit. Address as Sir. "
                "If idle: something like 'All systems nominal' or 'Awaiting your command.'"
            )
            text = llm_invoke(prompt)
            if text and text.strip():
                tts_speak(text.strip())
            else:
                tts_speak(f"Sir, you have {summary}." if "idle" not in summary else "Sir, all systems nominal. Awaiting your command.")
        except Exception:
            tts_speak(f"Sir, you have {summary}." if "idle" not in summary else "Sir, all systems nominal.")
    else:
        tts_speak(f"Sir, you have {summary}." if "idle" not in summary else "Sir, all systems nominal. Awaiting your command.")


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
