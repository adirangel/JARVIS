"""Audio-to-text transcription. Wraps voice.stt (faster-whisper) for compatibility."""

from __future__ import annotations

_stt_instance = None


def transcribe(
    audio_path: str,
    api_key: str | None = None,
    initial_prompt: str | None = None,
) -> str:
    """Transcribe audio file to text.
    Uses local faster-whisper; api_key is ignored (kept for API compatibility).
    Pinned to English for better accuracy with wake word detection.
    initial_prompt biases Whisper toward expected phrases (e.g. "Hey Jarvis").
    """
    global _stt_instance
    if _stt_instance is None:
        from voice.stt import SpeechToText

        _stt_instance = SpeechToText(
            model_name="large-v3-turbo",
            device="cuda",
            beam_size=3,
            compute_type="int8",
            language="en",
        )
    return _stt_instance.transcribe(audio_path, language="en", initial_prompt=initial_prompt)
