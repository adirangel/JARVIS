"""Text-to-Speech using Piper (English) + edge-tts (Hebrew).

Piper: British male (jgkawell/jarvis). Hebrew: edge-tts he-IL-AvriNeural.
"""

from __future__ import annotations

import asyncio
import re
import tempfile
import wave
from pathlib import Path
from typing import Optional

try:
    from piper import PiperVoice
    from huggingface_hub import hf_hub_download
    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False
    PiperVoice = None

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

_HEBREW_RE = re.compile(r"[\u0590-\u05FF]")


def _has_hebrew(text: str) -> bool:
    return bool(_HEBREW_RE.search(text))


def _rtl_display(text: str) -> str:
    """Convert Hebrew/RTL to visual order for correct display in LTR terminals (right-to-left read)."""
    if not text or not _has_hebrew(text):
        return text
    try:
        from bidi import get_display
        return get_display(text, base_dir="R")
    except ImportError:
        return text


class PiperTTS:
    """Piper TTS with jgkawell/jarvis British male voice."""

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
    """Piper for English, edge-tts for Hebrew (correct RTL pronunciation)."""

    def __init__(self, quality: str = "medium", hebrew_voice: str = "he-IL-AvriNeural", speed: float = 1.15):
        self._piper = PiperTTS(quality=quality, length_scale=1.0 / speed) if PIPER_AVAILABLE else None
        self._hebrew_voice = hebrew_voice
        self._speed = speed

    def synthesize(self, text: str, output_path: Optional[str] = None) -> str:
        if not text or not text.strip():
            raise ValueError("No text for synthesis")
        if output_path is None:
            f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=tempfile.gettempdir())
            output_path = f.name
            f.close()

        if _has_hebrew(text) and EDGE_TTS_AVAILABLE:
            self._synthesize_hebrew(text, output_path)
        else:
            self._synthesize_piper(text, output_path)
        return output_path

    def _synthesize_hebrew(self, text: str, output_path: str) -> None:
        """Use edge-tts for Hebrew (correct RTL, native pronunciation)."""
        pct = int((self._speed - 1) * 100)
        rate = f"+{pct}%" if pct >= 0 else f"{pct}%"
        comm = edge_tts.Communicate(text, self._hebrew_voice, rate=rate)
        comm.save_sync(output_path)

    def _synthesize_piper(self, text: str, output_path: str) -> None:
        """Use Piper for English."""
        if not self._piper:
            raise ImportError("piper-tts required for English. pip install piper-tts")
        self._piper.synthesize(text, output_path=output_path)


def create_tts(
    engine: str = "piper",
    voice: str = "jgkawell/jarvis",
    quality: str = "medium",
    hebrew_voice: Optional[str] = None,
    speed: float = 1.15,
):
    """Create TTS. Uses Piper (English) + edge-tts (Hebrew when detected). speed>1 = faster."""
    return HybridTTS(quality=quality, hebrew_voice=hebrew_voice or "he-IL-AvriNeural", speed=speed)
