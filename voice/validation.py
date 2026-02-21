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
    "תודה", "בבקשה", "כן", "לא", "אוקיי",  # Hebrew common artifacts
})

# Minimum word count - single words often false positives
_DEFAULT_MIN_WORDS = 3

logger = logging.getLogger(__name__)


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
