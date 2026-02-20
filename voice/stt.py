"""Speech-to-Text using faster-whisper (large-v3-turbo).

GPU accelerated, excellent Hebrew + English detection.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

# Ensure CUDA 12 DLLs are findable on Windows (cublas64_12.dll)
if sys.platform == "win32":
    _cuda_bin = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    for v in ("v12.6", "v12.4", "v12.8"):
        d = _cuda_bin / v / "bin"
        if d.is_dir() and (d / "cublas64_12.dll").exists():
            d_str = str(d)
            os.add_dll_directory(d_str)
            # Also prepend to PATH so child processes and loaders find it
            os.environ["PATH"] = d_str + os.pathsep + os.environ.get("PATH", "")
            break

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    WhisperModel = None


class SpeechToText:
    """STT via faster-whisper large-v3-turbo."""

    def __init__(
        self,
        model_name: str = "large-v3-turbo",
        device: str = "cuda",
        language: Optional[str] = None,
    ):
        if not FASTER_WHISPER_AVAILABLE:
            raise ImportError("faster-whisper required. pip install faster-whisper")
        self._model_name = model_name
        self._device = device
        self._language = language  # None = auto-detect
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is None:
            device = self._device
            compute_type = "float16" if device == "cuda" else "int8"
            try:
                self._model = WhisperModel(
                    self._model_name,
                    device=device,
                    compute_type=compute_type,
                )
            except Exception as e:
                if device == "cuda" and ("cublas" in str(e).lower() or "cuda" in str(e).lower()):
                    print("CUDA unavailable, falling back to CPU for STT.", flush=True)
                    self._model = WhisperModel(
                        self._model_name,
                        device="cpu",
                        compute_type="int8",
                    )
                else:
                    raise

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
    ) -> str:
        """Transcribe audio file to text."""
        self._ensure_model()
        lang = language or self._language
        try:
            segments, info = self._model.transcribe(
                audio_path,
                language=lang,
                beam_size=5,
                vad_filter=False,  # VAD can sometimes filter out valid speech
            )
        except RuntimeError as e:
            if self._device == "cuda" and ("cublas" in str(e).lower() or "cuda" in str(e).lower() or "dll" in str(e).lower()):
                print("CUDA failed at inference, retrying with CPU...", flush=True)
                self._model = None
                self._device = "cpu"
                self._ensure_model()
                segments, info = self._model.transcribe(
                    audio_path,
                    language=lang,
                    beam_size=5,
                    vad_filter=False,
                )
            else:
                raise
        text = " ".join(s.text for s in segments).strip()
        return text or ""

    def transcribe_bytes(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        """Transcribe raw audio bytes (mono, 16kHz typical)."""
        import tempfile
        import wave
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            with wave.open(f.name, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                wav.writeframes(audio_bytes)
            return self.transcribe(f.name)
