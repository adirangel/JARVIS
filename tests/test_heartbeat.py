"""Tests for heartbeat."""

import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_heartbeat_job():
    from heartbeat import heartbeat_job

    class MockMemory:
        def get_pending_tasks(self):
            return [{"id": 1, "description": "Test"}]
        def get_reminders(self):
            return []
        def mark_reminder_done(self, rid):
            pass

    spoken = []
    def mock_tts(text):
        spoken.append(text)

    heartbeat_job(MockMemory(), mock_tts, llm_invoke=None)
    assert len(spoken) == 1
    assert "task" in spoken[0].lower() or "pending" in spoken[0].lower()


def test_heartbeat_job_idle():
    """Heartbeat speaks even when idle (no pending tasks/reminders)."""
    from heartbeat import heartbeat_job

    class MockMemory:
        def get_pending_tasks(self):
            return []
        def get_reminders(self):
            return []

    spoken = []
    def mock_tts(text):
        spoken.append(text)

    heartbeat_job(MockMemory(), mock_tts, llm_invoke=None)
    assert len(spoken) == 1
    assert "Sir" in spoken[0]
    assert "nominal" in spoken[0].lower() or "awaiting" in spoken[0].lower()
