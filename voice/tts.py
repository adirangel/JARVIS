"""Text-to-Speech using Piper (jgkawell/jarvis British male voice)."""

from __future__ import annotations

import re
import tempfile
import wave
from pathlib import Path
from typing import Optional

try:
    import numpy as np
    import soundfile as sf
    SF_AVAILABLE = True
except ImportError:
    SF_AVAILABLE = False
    np = None
    sf = None

try:
    from piper import PiperVoice
    from huggingface_hub import hf_hub_download
    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False
    PiperVoice = None

_SENTENCE_BREAK_RE = re.compile(r"(?<=[\.\!\?;:\n])\s+")


def _rtl_display(text: str) -> str:
    """Pass-through for display (kept for API compatibility)."""
    return text or ""


def _normalize_text(text: str) -> str:
    return " ".join((text or "").replace("\r", " ").replace("\n", " ").split()).strip()


def _split_tts_chunks(text: str, max_chars: int = 280) -> list[str]:
    """Split long text into sentence chunks to avoid truncation."""
    normalized = _normalize_text(text)
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized]

    pieces: list[str] = []
    parts = _SENTENCE_BREAK_RE.split(normalized)
    current: list[str] = []
    cur_len = 0

    for part in parts:
        p = part.strip()
        if not p:
            continue
        if len(p) > max_chars:
            words = p.split()
            buf: list[str] = []
            blen = 0
            for w in words:
                extra = len(w) + (1 if buf else 0)
                if blen + extra > max_chars and buf:
                    pieces.append(" ".join(buf))
                    buf = [w]
                    blen = len(w)
                else:
                    buf.append(w)
                    blen += extra
            if buf:
                pieces.append(" ".join(buf))
            continue

        extra = len(p) + (1 if current else 0)
        if cur_len + extra > max_chars and current:
            pieces.append(" ".join(current))
            current = [p]
            cur_len = len(p)
        else:
            current.append(p)
            cur_len += extra

    if current:
        pieces.append(" ".join(current))

    return [c.strip() for c in pieces if c.strip()]


def _with_interruption_note(text: str) -> str:
    t = (text or "").rstrip()
    if not t:
        return t
    if t.endswith((".", "!", "?", "...")):
        return t
    return f"{t} Pardon the interruption, Sir."


class PiperTTS:
    """Piper TTS with jgkawell/jarvis British male voice (JARVIS)."""

    JARVIS_REPO = "jgkawell/jarvis"
    MEDIUM = ("en/en_GB/jarvis/medium/jarvis-medium.onnx", "en/en_GB/jarvis/medium/jarvis-medium.onnx.json")
    HIGH = ("en/en_GB/jarvis/high/jarvis-high.onnx", "en/en_GB/jarvis/high/jarvis-high.onnx.json")

    def __init__(self, quality: str = "medium", length_scale: Optional[float] = None):
        if not PIPER_AVAILABLE:
            raise ImportError("piper-tts and huggingface_hub required. pip install piper-tts huggingface_hub")
        model_path, config_path = self.HIGH if quality == "high" else self.MEDIUM
        m = hf_hub_download(self.JARVIS_REPO, model_path)
        c = hf_hub_download(self.JARVIS_REPO, config_path)
        self._voice = PiperVoice.load(m, config_path=c)
        self._length_scale = length_scale

    def synthesize(self, text: str, output_path: Optional[str] = None) -> str:
        if not text or not text.strip():
            raise ValueError("No text for synthesis")
        if output_path is None:
            f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=tempfile.gettempdir())
            output_path = f.name
            f.close()
        syn_config = None
        if self._length_scale is not None:
            from piper.config import SynthesisConfig
            syn_config = SynthesisConfig(length_scale=self._length_scale)
        with wave.open(output_path, "wb") as wav:
            self._voice.synthesize_wav(text, wav, syn_config=syn_config)
        return output_path

    @staticmethod
    def is_available() -> bool:
        return PIPER_AVAILABLE


class HybridTTS:
    """English-only TTS with chunked synthesis for long responses."""

    def __init__(self, quality: str = "medium", speed: float = 1.15):
        self._piper = PiperTTS(quality=quality, length_scale=1.0 / speed) if PIPER_AVAILABLE else None

    def synthesize(self, text: str, output_path: Optional[str] = None, language_hint: Optional[str] = None) -> str:
        if not text or not text.strip():
            raise ValueError("No text for synthesis")
        out = output_path or _tmp_audio_path()
        chunks = _split_tts_chunks(text)
        if not chunks:
            raise ValueError("No text chunks for synthesis")

        try:
            self._synthesize_chunks(chunks, out)
            if not _is_audio_valid(out):
                raise RuntimeError("Generated audio seems incomplete")
            return out
        except Exception:
            retry = _with_interruption_note(text)
            self._synthesize_chunks(_split_tts_chunks(retry), out)
            return out

    def _synthesize_chunks(self, chunks: list[str], output_path: str) -> None:
        temp_paths: list[str] = []
        try:
            for chunk in chunks:
                temp_out = _tmp_audio_path()
                if self._piper is None:
                    raise ImportError("piper-tts required. pip install piper-tts")
                self._piper.synthesize(chunk, output_path=temp_out)
                temp_paths.append(temp_out)
            _concat_wavs(temp_paths, output_path)
        finally:
            for p in temp_paths:
                try:
                    Path(p).unlink(missing_ok=True)
                except Exception:
                    pass


def _tmp_audio_path() -> str:
    f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=tempfile.gettempdir())
    path = f.name
    f.close()
    return path


def _concat_wavs(paths: list[str], output_path: str) -> None:
    if not paths:
        raise ValueError("No audio chunks to concatenate")
    if len(paths) == 1:
        Path(paths[0]).replace(output_path)
        return

    if SF_AVAILABLE:
        data_all = []
        sr0: Optional[int] = None
        for p in paths:
            data, sr = sf.read(p)
            if sr0 is None:
                sr0 = sr
            if sr != sr0:
                raise ValueError("Mismatched sample rates across TTS chunks")
            data_all.append(data)
        merged = np.concatenate(data_all)
        sf.write(output_path, merged, sr0)
        return

    params = None
    raw_frames = []
    for p in paths:
        with wave.open(p, "rb") as wav:
            if params is None:
                params = wav.getparams()
            elif wav.getparams()[:4] != params[:4]:
                raise ValueError("Mismatched WAV params across TTS chunks")
            raw_frames.append(wav.readframes(wav.getnframes()))
    with wave.open(output_path, "wb") as out:
        out.setparams(params)
        for fr in raw_frames:
            out.writeframes(fr)


def _is_audio_valid(path: str) -> bool:
    p = Path(path)
    if not p.exists() or p.stat().st_size <= 512:
        return False
    if SF_AVAILABLE:
        try:
            info = sf.info(path)
            return info.frames > 0 and info.samplerate > 0
        except Exception:
            return False
    return True


def create_tts(
    engine: str = "piper",
    voice: str = "jgkawell/jarvis",
    quality: str = "medium",
    speed: float = 1.15,
    preload: bool = False,
):
    """Create TTS. Piper jgkawell/jarvis (JARVIS voice). preload=warm up at init."""
    tts = HybridTTS(quality=quality, speed=speed)
    if preload:
        try:
            tts.synthesize("As you wish, Sir.")
        except Exception:
            pass
    return tts
