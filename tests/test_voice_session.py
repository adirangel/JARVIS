"""Tests for voice active-session state machine."""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_session_activate_timeout_and_end():
    from voice.session import VoiceSession

    s = VoiceSession(silence_timeout=0.08)
    assert s.is_active() is False
    s.activate()
    assert s.is_active() is True
    assert s.timed_out() is False
    time.sleep(0.12)
    assert s.timed_out() is True
    s.end()
    assert s.is_active() is False


def test_end_command_detection():
    from voice.session import contains_end_command

    assert contains_end_command("goodbye for now") is True
    assert contains_end_command("please stop listening") is True
    assert contains_end_command("let us continue") is False
