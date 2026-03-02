"""TTS module - Piper (WAV) or edge-tts (MP3). Piper uses sounddevice for playback (no ffmpeg)."""

import asyncio
import io
import re
import tempfile
import os
from typing import Optional


def _sanitize_for_speech(text: str) -> str:
    """Remove emojis, asterisks, quotation marks - real speech uses tone, not formatting."""
    if not text:
        return text
    # Remove emoji (Unicode ranges)
    text = re.sub(r'[\U0001F300-\U0001F9FF]', '', text)  # Misc symbols, emoji
    text = re.sub(r'[\u2600-\u26FF\u2700-\u27BF]', '', text)  # Misc symbols
    # Remove asterisks (bold/italic markers)
    text = re.sub(r'\*+', '', text)
    # Remove double quotes only (keep apostrophes for contractions like "don't")
    text = re.sub(r'["""]', '', text)
    return text.strip()

# Legacy function for backward compatibility
def text_to_speech(api_key, region, text, output_file_path):
    from azure.cognitiveservices.speech import SpeechConfig, SpeechSynthesizer, AudioConfig, SpeechSynthesisOutputFormat, ResultReason
    from azure.cognitiveservices.speech.audio import AudioOutputConfig

    speech_config = SpeechConfig(subscription=api_key, region=region)
    speech_config.set_speech_synthesis_output_format(SpeechSynthesisOutputFormat["Riff16Khz16BitMonoPcm"])
    speech_config.speech_synthesis_voice_name = "en-US-JessaNeural"
    audio_output_config = AudioOutputConfig(filename=output_file_path)
    synthesizer = SpeechSynthesizer(speech_config=speech_config, audio_config=audio_output_config)
    result = synthesizer.speak_text_async(text).get()
    if result.reason == ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print(f"Speech synthesis canceled: {cancellation_details.reason}")
    else:
        print("Text to speech conversion successful!")


class TTS:
    """Streaming TTS - Piper (WAV, no ffmpeg) or edge-tts (MP3)."""

    def __init__(self, config: dict):
        self.config = config
        self.engine = (config.get("tts_engine") or "piper").lower()
        self.voice_id = config.get("voice_id", "en-GB-SoniaNeural")
        self.speed = config.get("tts_speed", 1.0)
        self._buffer = []
        self._piper = None
        self._tts_lock = None  # Created on first use (avoids event loop issues at init)

    async def initialize(self) -> None:
        """Initialize TTS engine."""
        if self.engine == "piper":
            try:
                from voice.tts import PiperTTS
                quality = self.config.get("tts_quality", "medium")
                self._piper = PiperTTS(quality=quality, length_scale=1.0 / self.speed)
            except Exception:
                self._piper = None

    async def stream_chunk(self, text: str, state=None) -> None:
        """Buffer for TTS. Play on sentence boundaries in background - do NOT block text streaming."""
        if not text or not text.strip():
            return
        self._buffer.append(text)
        full = "".join(self._buffer)
        stripped = full.strip()
        # When we have a complete sentence, schedule playback (non-blocking)
        if stripped and (stripped[-1] in ".!?" or "\n" in full):
            if state is None or not self._is_stopped(state):
                to_play = full.strip()
                if to_play:
                    self._buffer = []
                    if self._tts_lock is None:
                        self._tts_lock = asyncio.Lock()
                    async def _play_in_background():
                        async with self._tts_lock:
                            if state is not None and self._is_stopped(state):
                                return
                            await self._synthesize_and_play(to_play, state)
                    asyncio.create_task(_play_in_background())

    async def finalize(self, state=None) -> None:
        """Synthesize buffered text and play. Waits for any in-flight playback, then plays remainder."""
        if state is not None and self._is_stopped(state):
            self._buffer = []
            return
        full_text = "".join(self._buffer).strip()
        self._buffer = []
        if not full_text:
            return
        if self._tts_lock is None:
            self._tts_lock = asyncio.Lock()
        async with self._tts_lock:
            if state is not None and self._is_stopped(state):
                return
            await self._synthesize_and_play(full_text, state)

    async def _synthesize_and_play(self, text: str, state=None) -> None:
        """Synthesize and play - Piper (WAV) or edge-tts (MP3)."""
        text = _sanitize_for_speech(text)
        if not text:
            return
        if self.engine == "piper" and self._piper:
            await self._play_piper(text, state)
        else:
            await self._play_edge_tts(text, state)

    async def _play_piper(self, text: str, state=None) -> None:
        """Piper TTS -> WAV -> sounddevice (no ffmpeg)."""
        try:
            wav_path = self._piper.synthesize(text)
            await self._play_wav_file(wav_path, state)
            try:
                os.unlink(wav_path)
            except OSError:
                pass
        except Exception as e:
            print(f"[TTS] Piper failed: {e}, falling back to edge-tts")
            await self._play_edge_tts(text, state)

    def _is_stopped(self, state) -> bool:
        return state is not None and getattr(state, "stopped", None) and state.stopped.is_set()

    async def _play_wav_file(self, path: str, state=None) -> None:
        """Play WAV using sounddevice. Interruptible via state.stopped."""
        import sounddevice as sd
        import soundfile as sf
        data, samplerate = sf.read(path, dtype="float32")
        sd.play(data, samplerate, blocking=False)
        # sd.play() returns None on Windows when blocking=False; use duration-based wait
        duration_sec = len(data) / float(samplerate)
        elapsed = 0.0
        while elapsed < duration_sec:
            if self._is_stopped(state):
                sd.stop()
                return
            await asyncio.sleep(0.05)
            elapsed += 0.05

    async def _play_edge_tts(self, text: str, state=None) -> None:
        """Edge-TTS -> MP3. Play via sounddevice if possible, else pydub."""
        import edge_tts
        voice = self.voice_id if self.voice_id != "default" else "en-GB-SoniaNeural"
        rate = f"{int((self.speed - 1) * 100)}%" if self.speed != 1.0 else "+0%"
        communicate = edge_tts.Communicate(text, voice, rate=rate)
        chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                chunks.append(chunk["data"])
        if not chunks:
            return
        data = b"".join(chunks)
        await self._play_mp3(data, state)

    async def _play_mp3(self, data: bytes, state=None) -> None:
        """Play MP3 via pydub (blocking, less interruptible). Piper path is preferred."""
        try:
            from pydub import AudioSegment
            from pydub.playback import play
            audio = AudioSegment.from_mp3(io.BytesIO(data))
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: play(audio))
        except Exception:
            print("[TTS] Install ffmpeg for edge-tts voice: pip install ffmpeg-python, or use tts_engine: piper")
