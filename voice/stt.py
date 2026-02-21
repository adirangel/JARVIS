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
    """STT via faster-whisper large-v3-turbo. int8 + beam_size=3 for sub-2s latency."""

    def __init__(
        self,
        model_name: str = "large-v3-turbo",
        device: str = "cuda",
        language: Optional[str] = None,
        beam_size: int = 3,
        compute_type: Optional[str] = None,
    ):
        if not FASTER_WHISPER_AVAILABLE:
            raise ImportError("faster-whisper required. pip install faster-whisper")
        self._model_name = model_name
        self._device = device
        self._language = language
        self._beam_size = beam_size
        self._compute_type = compute_type
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is None:
            device = self._device
            compute_type = self._compute_type
            if compute_type is None:
                compute_type = "int8" if device == "cuda" else "int8"  # int8 = faster on both
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
        vad_filter: Optional[bool] = None,
    ) -> str:
        """Transcribe audio file to text.

        vad_filter: Use Silero VAD to filter silence/background (reduces false positives).
        Default True when not specified (config voice.use_vad).
        """
        self._ensure_model()
        lang = language or self._language
        # Issue 1: VAD filters silence/background - reduces "thank you" etc. from noise
        use_vad = vad_filter if vad_filter is not None else True
        try:
            segments, info = self._model.transcribe(
                audio_path,
                language=lang,
                beam_size=self._beam_size,
                vad_filter=use_vad,
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
                    beam_size=self._beam_size,
                    vad_filter=use_vad,
                )
            else:
                raise
        text = " ".join(s.text for s in segments).strip()
        return text or ""

    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        sample_rate: int = 16000,
        vad_filter: Optional[bool] = None,
    ) -> str:
        """Transcribe raw audio bytes (mono, 16kHz typical)."""
        import tempfile
        import wave
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            with wave.open(f.name, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                wav.writeframes(audio_bytes)
            return self.transcribe(f.name, vad_filter=vad_filter)
