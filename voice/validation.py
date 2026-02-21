"""Post-STT validation to reduce false positives from noise/mis-transcription.

Discards empty, short, or common noise artifacts (e.g. "thank you" from background).
Logs rejected transcriptions for debugging.
"""

from __future__ import annotations

import logging
from typing import Optional

# Common noise artifacts - Whisper often hallucinates these from silence/background
_NOISE_ARTIFACTS = frozenset({
    "thank you", "thanks", "thank", "you", "bye", "goodbye", "noise",
    "no", "yes", "ok", "okay", "um", "uh", "hmm", "ah", "oh",
    "the", "a", "an", "and", "or", "but", "so", "to", "of", "in",
})

# Minimum word count - single words often false positives
_DEFAULT_MIN_WORDS = 3

logger = logging.getLogger(__name__)

try:
    import webrtcvad
    WEBRTC_VAD_AVAILABLE = True
except ImportError:
    WEBRTC_VAD_AVAILABLE = False
    webrtcvad = None


def is_valid_transcript(
    text: str,
    min_words: int = _DEFAULT_MIN_WORDS,
    noise_artifacts: Optional[frozenset] = None,
    log_rejections: bool = True,
) -> tuple[bool, str]:
    """Validate transcribed text. Returns (is_valid, reason).

    Rejects: empty, too short (< min_words), or known noise artifacts.
    Logs false positives when log_rejections=True for debugging.
    """
    artifacts = noise_artifacts or _NOISE_ARTIFACTS
    t = (text or "").strip()
    if not t:
        if log_rejections:
            logger.debug("[STT validation] Rejected: empty transcript (false positive)")
        return False, "empty"

    words = t.lower().split()
    if len(words) < min_words:
        if log_rejections:
            logger.debug("[STT validation] Rejected: too short (%d words) - %r", len(words), t[:80])
        return False, "too_short"

    # Exact match or single-word artifact
    t_lower = t.lower()
    if t_lower in artifacts:
        if log_rejections:
            logger.debug("[STT validation] Rejected: noise artifact - %r", t)
        return False, "noise_artifact"

    # Single word that's a known artifact
    if len(words) == 1 and words[0] in artifacts:
        if log_rejections:
            logger.debug("[STT validation] Rejected: single-word artifact - %r", t)
        return False, "noise_artifact"

    return True, ""


def has_voice_activity(
    audio_bytes: bytes,
    sample_rate: int = 16000,
    min_speech_ratio: float = 0.08,
    vad_aggressiveness: int = 2,
) -> tuple[bool, str]:
    """Pre-STT VAD gate to reject background noise before Whisper.

    Uses WebRTC VAD when available; falls back to RMS-only acceptance.
    """
    if not audio_bytes:
        return False, "empty_audio"

    if not WEBRTC_VAD_AVAILABLE:
        return True, "vad_unavailable"

    try:
        if sample_rate not in (8000, 16000, 32000, 48000):
            return True, "unsupported_sample_rate"

        frame_ms = 30
        frame_size = int(sample_rate * frame_ms / 1000) * 2  # int16 bytes
        if frame_size <= 0 or len(audio_bytes) < frame_size:
            return False, "audio_too_short_for_vad"

        vad = webrtcvad.Vad(max(0, min(int(vad_aggressiveness), 3)))
        total = 0
        speech = 0
        for i in range(0, len(audio_bytes) - frame_size + 1, frame_size):
            frame = audio_bytes[i : i + frame_size]
            total += 1
            if vad.is_speech(frame, sample_rate):
                speech += 1

        if total == 0:
            return False, "no_frames"

        ratio = speech / float(total)
        return ratio >= float(min_speech_ratio), f"speech_ratio={ratio:.3f}"
    except Exception as e:
        logger.debug("[STT validation] VAD error: %s", e)
        return True, "vad_error_fallback"
