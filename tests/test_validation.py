"""Tests for voice validation (false positive reduction)."""

import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_valid_transcript_accepted():
    from voice.validation import is_valid_transcript
    valid, _ = is_valid_transcript("What is the time in Tokyo?")
    assert valid is True
    valid, _ = is_valid_transcript("Thank you so much for your help")
    assert valid is True


def test_empty_rejected():
    from voice.validation import is_valid_transcript
    valid, reason = is_valid_transcript("")
    assert valid is False
    assert reason == "empty"
    valid, reason = is_valid_transcript("   ")
    assert valid is False
    assert reason == "empty"


def test_too_short_rejected():
    from voice.validation import is_valid_transcript
    valid, reason = is_valid_transcript("hi", min_words=3)
    assert valid is False
    assert reason == "too_short"
    valid, reason = is_valid_transcript("thank you", min_words=3)
    assert valid is False
    assert reason == "too_short"


def test_allowed_short_commands():
    """Short commands like 'stop', 'goodbye' should be accepted even with min_words=3."""
    from voice.validation import is_valid_transcript
    valid, _ = is_valid_transcript("stop", min_words=3, allowed_short=["stop", "goodbye", "exit"])
    assert valid is True
    valid, _ = is_valid_transcript("goodbye", min_words=3, allowed_short=["stop", "goodbye"])
    assert valid is True
    valid, _ = is_valid_transcript("please stop", min_words=3, allowed_short=["stop", "goodbye"])
    assert valid is True


def test_noise_artifact_rejected():
    from voice.validation import is_valid_transcript
    valid, reason = is_valid_transcript("thank you", min_words=2)
    assert valid is False
    assert reason == "noise_artifact"
    valid, reason = is_valid_transcript("noise", min_words=1)
    assert valid is False
    assert reason == "noise_artifact"


def test_ensure_complete_sentence():
    from agent.graph import _ensure_complete_sentence
    assert _ensure_complete_sentence("Hello, Sir.") == "Hello, Sir."
    assert _ensure_complete_sentence("As you wish") == "As you wish Pardon the interruption, Sir."
    assert _ensure_complete_sentence("") == ""


def test_has_voice_activity_empty_audio_rejected():
    from voice.validation import has_voice_activity

    ok, reason = has_voice_activity(b"", sample_rate=16000)
    assert ok is False
    assert reason == "empty_audio"
