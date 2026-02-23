"""Audio-to-text transcription. Wraps voice.stt (faster-whisper) for compatibility."""

from __future__ import annotations

_stt_instance = None


def transcribe(audio_path: str, api_key: str | None = None) -> str:
    """Transcribe audio file to text.
    Uses local faster-whisper; api_key is ignored (kept for API compatibility).
    """
    global _stt_instance
    if _stt_instance is None:
        from voice.stt import SpeechToText

        _stt_instance = SpeechToText(
            model_name="large-v3-turbo",
            device="cuda",
            beam_size=3,
            compute_type="int8",
        )
    return _stt_instance.transcribe(audio_path)
