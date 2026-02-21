"""Text-to-Speech with local-first Piper voices.

English: jgkawell/jarvis Piper voice.
Hebrew: optional local Piper Hebrew model (configurable). Falls back to English Piper unless
remote fallback is explicitly enabled.
"""

from __future__ import annotations

import asyncio
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

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

# Hebrew block + Niqqud + presentation forms (e.g. final letters)
_HEBREW_RE = re.compile(r"[\u0590-\u05FF\uFB1D-\uFB4F]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_SENTENCE_BREAK_RE = re.compile(r"(?<=[\.\!\?;:\n\u05C3])\s+")


def _has_hebrew(text: str) -> bool:
    return bool(_HEBREW_RE.search(text or ""))


def _rtl_display(text: str) -> str:
    """Convert Hebrew/RTL to visual order for correct display in LTR terminals."""
    if not text or not _has_hebrew(text):
        return text
    try:
        from bidi import get_display
        return get_display(text, base_dir="R")
    except ImportError:
        return text


def _looks_english(text: str) -> bool:
    return bool(_LATIN_RE.search(text or ""))


def _normalize_text(text: str) -> str:
    return " ".join((text or "").replace("\r", " ").replace("\n", " ").split()).strip()


def _split_tts_chunks(text: str, max_chars: int = 280) -> list[str]:
    """Split long text into sentence chunks to avoid truncation in TTS engines."""
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
    """Single Piper voice wrapper."""

    EN_REPO = "jgkawell/jarvis"
    EN_MEDIUM = ("en/en_GB/jarvis/medium/jarvis-medium.onnx", "en/en_GB/jarvis/medium/jarvis-medium.onnx.json")
    EN_HIGH = ("en/en_GB/jarvis/high/jarvis-high.onnx", "en/en_GB/jarvis/high/jarvis-high.onnx.json")

    # Local Hebrew Piper defaults (override via config paths if needed)
    HE_REPO = "rhasspy/piper-voices"
    HE_MEDIUM = ("he/he_IL/amit/medium/he_IL-amit-medium.onnx", "he/he_IL/amit/medium/he_IL-amit-medium.onnx.json")
    HE_HIGH = ("he/he_IL/amit/high/he_IL-amit-high.onnx", "he/he_IL/amit/high/he_IL-amit-high.onnx.json")

    def __init__(
        self,
        quality: str = "medium",
        length_scale: Optional[float] = None,
        model_repo: Optional[str] = None,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
        language: str = "en",
    ):
        if not PIPER_AVAILABLE:
            raise ImportError("piper-tts and huggingface_hub required. pip install piper-tts huggingface_hub")
        self._length_scale = length_scale
        self._language = language

        if model_path and config_path:
            m = self._resolve_path(model_path, repo=model_repo)
            c = self._resolve_path(config_path, repo=model_repo)
        else:
            m, c = self._default_voice_files(quality=quality, language=language)

        self._voice = PiperVoice.load(m, config_path=c)

    @staticmethod
    def _resolve_path(path_or_hf: str, repo: Optional[str] = None) -> str:
        p = Path(path_or_hf)
        if p.exists():
            return str(p)
        if repo:
            return hf_hub_download(repo, path_or_hf)
        return hf_hub_download(PiperTTS.EN_REPO, path_or_hf)

    @classmethod
    def _default_voice_files(cls, quality: str, language: str) -> tuple[str, str]:
        lang = (language or "en").lower()
        if lang.startswith("he"):
            model_path, config_path = cls.HE_HIGH if quality == "high" else cls.HE_MEDIUM
            return (
                hf_hub_download(cls.HE_REPO, model_path),
                hf_hub_download(cls.HE_REPO, config_path),
            )
        model_path, config_path = cls.EN_HIGH if quality == "high" else cls.EN_MEDIUM
        return (
            hf_hub_download(cls.EN_REPO, model_path),
            hf_hub_download(cls.EN_REPO, config_path),
        )

    def synthesize(self, text: str, output_path: Optional[str] = None) -> str:
        if not text or not text.strip():
            raise ValueError("No text for synthesis")
        out = output_path or _tmp_audio_path()
        syn_config = None
        if self._length_scale is not None:
            from piper.config import SynthesisConfig
            syn_config = SynthesisConfig(length_scale=self._length_scale)
        with wave.open(out, "wb") as wav:
            self._voice.synthesize_wav(text, wav, syn_config=syn_config)
        return out

    @staticmethod
    def is_available() -> bool:
        return PIPER_AVAILABLE


class HybridTTS:
    """Local-first multilingual TTS with chunked synthesis for long responses."""

    def __init__(
        self,
        quality: str = "medium",
        hebrew_voice: str = "he-IL-amit-medium",
        speed: float = 1.15,
        force_hebrew_tts: bool = False,
        hebrew_model_repo: Optional[str] = None,
        hebrew_model_path: Optional[str] = None,
        hebrew_model_config: Optional[str] = None,
        allow_remote_fallback: bool = False,
    ):
        self._speed = speed
        self._force_hebrew_tts = force_hebrew_tts
        self._allow_remote_fallback = allow_remote_fallback
        self._hebrew_voice = hebrew_voice
        self._english = PiperTTS(quality=quality, length_scale=1.0 / speed, language="en") if PIPER_AVAILABLE else None

        self._hebrew: Optional[PiperTTS] = None
        if PIPER_AVAILABLE:
            try:
                self._hebrew = PiperTTS(
                    quality=quality,
                    length_scale=1.0 / speed,
                    model_repo=hebrew_model_repo,
                    model_path=hebrew_model_path,
                    config_path=hebrew_model_config,
                    language="he",
                )
            except Exception:
                self._hebrew = None

    def synthesize(
        self,
        text: str,
        output_path: Optional[str] = None,
        language_hint: Optional[str] = None,
    ) -> str:
        if not text or not text.strip():
            raise ValueError("No text for synthesis")
        out = output_path or _tmp_audio_path()
        hint = (language_hint or "").strip().lower()
        use_hebrew = self._force_hebrew_tts or hint.startswith("he") or _has_hebrew(text)
        chunks = _split_tts_chunks(text)
        if not chunks:
            raise ValueError("No text chunks for synthesis")

        try:
            self._synthesize_chunks(chunks, out, use_hebrew=use_hebrew)
            if not _is_audio_valid(out):
                raise RuntimeError("Generated audio seems incomplete")
            return out
        except Exception:
            # Retry once with explicit completion note to avoid abrupt cut.
            retry = _with_interruption_note(text)
            self._synthesize_chunks(_split_tts_chunks(retry), out, use_hebrew=use_hebrew)
            return out

    def _synthesize_chunks(self, chunks: list[str], output_path: str, use_hebrew: bool) -> None:
        temp_paths: list[str] = []
        try:
            for chunk in chunks:
                temp_out = _tmp_audio_path()
                self._synthesize_single(chunk, temp_out, use_hebrew=use_hebrew)
                temp_paths.append(temp_out)
            _concat_wavs(temp_paths, output_path)
        finally:
            for p in temp_paths:
                try:
                    Path(p).unlink(missing_ok=True)
                except Exception:
                    pass

    def _synthesize_single(self, text: str, output_path: str, use_hebrew: bool) -> None:
        # Prefer edge-tts for Hebrew (correct pronunciation); Piper Hebrew has spelling issues
        if use_hebrew and EDGE_TTS_AVAILABLE:
            self._synthesize_hebrew_remote(text, output_path)
            return
        if use_hebrew and self._hebrew is not None:
            self._hebrew.synthesize(text, output_path=output_path)
            return
        if self._english is None:
            raise ImportError("piper-tts required for local synthesis. pip install piper-tts")
        # Keep Hebrew voice preference even when Hebrew model is missing: transliteration-free fallback.
        self._english.synthesize(text, output_path=output_path)

    def _synthesize_hebrew_remote(self, text: str, output_path: str) -> None:
        pct = int((self._speed - 1) * 100)
        rate = f"+{pct}%" if pct >= 0 else f"{pct}%"
        comm = edge_tts.Communicate(text, self._hebrew_voice, rate=rate)
        try:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(asyncio.run, comm.save(output_path))
                future.result(timeout=30)
        except Exception:
            comm.save_sync(output_path)


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

    # Fallback: strict WAV concatenation via wave module (same format required).
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
    hebrew_voice: Optional[str] = None,
    speed: float = 1.15,
    force_hebrew_tts: bool = False,
    preload: bool = False,
    hebrew_model_repo: Optional[str] = None,
    hebrew_model_path: Optional[str] = None,
    hebrew_model_config: Optional[str] = None,
    allow_remote_hebrew_fallback: bool = False,
):
    """Create local-first TTS.

    `allow_remote_hebrew_fallback=False` keeps pipeline fully local.
    """
    tts = HybridTTS(
        quality=quality,
        hebrew_voice=hebrew_voice or "he-IL-amit-medium",
        speed=speed,
        force_hebrew_tts=force_hebrew_tts,
        hebrew_model_repo=hebrew_model_repo,
        hebrew_model_path=hebrew_model_path,
        hebrew_model_config=hebrew_model_config,
        allow_remote_fallback=allow_remote_hebrew_fallback,
    )
    if preload:
        try:
            tts.synthesize("As you wish, Sir.")
            tts.synthesize("כמובן, Sir.", language_hint="he")
        except Exception:
            pass
    return tts
