"""Wake-word listener with VAD gating, anti-hallucination, and mic muting.

Flow:
  1. Record 2-second audio chunks from the mic
  2. Pre-filter with WebRTC VAD (skip silence before calling Whisper)
  3. Transcribe with faster-whisper (initial_prompt="Hey Jarvis" for bias)
  4. Post-filter with validation (reject noise artifacts like "Thank you")
  5. Fuzzy-match for wake word ("Hey Jarvis" and common mishearings)
  6. During TTS playback, mute the transcription loop (prevent self-hearing)
"""

import threading
import time
import queue
import wave
import tempfile
import os
import re
from loguru import logger

try:
    import audio_to_text
    HAS_AUDIO_TO_TEXT = True
except ImportError:
    HAS_AUDIO_TO_TEXT = False

try:
    from voice.validation import is_valid_transcript, has_voice_activity
    HAS_VALIDATION = True
except ImportError:
    HAS_VALIDATION = False


class WakeListener:
    def __init__(self, config=None, wake_word="Hey Jarvis", debounce_seconds=1.5,
                 on_detected=None, verbose=False):
        if isinstance(config, dict):
            ww = config.get("wake_word", "Hey Jarvis")
            self.wake_word = (ww if isinstance(ww, str) else "Hey Jarvis").lower()
            self.debounce = config.get("debounce_seconds", 1.5)
            self.verbose = config.get("wake_verbose", False)
            self.api_key = config.get("stt_api_key") or config.get("llm_api_key")
            ww_cfg = config.get("wake_word", {}) if isinstance(config.get("wake_word"), dict) else {}
            self.input_device = config.get("input_device") or config.get("wake_device") or ww_cfg.get("device")
            self._session_timeout = float(config.get("session_idle_timeout_seconds", 600))
        else:
            self.wake_word = wake_word.lower()
            self.debounce = debounce_seconds
            self.verbose = verbose
            self.api_key = None
            self.input_device = None
            self._session_timeout = 600.0

        self.on_detected = on_detected
        self._running = False
        self._thread = None
        self._last_trigger = 0
        self._chunk_count = 0
        self._stt_loaded = False
        self._last_transcript = ""
        self._session_active = False
        self._last_speech_time = 0.0
        self.audio_queue = queue.Queue()

        # Mic muting: set when JARVIS is speaking to prevent self-hearing
        self._muted = threading.Event()

    # ── Mic muting ────────────────────────────────────────────────────────
    def mute(self):
        """Mute transcription (call when JARVIS starts speaking)."""
        self._muted.set()

    def unmute(self):
        """Unmute transcription (call when JARVIS finishes speaking)."""
        self._muted.clear()

    # ── Wake phrase stripping ─────────────────────────────────────────────
    def _strip_wake_phrase(self, text: str) -> str:
        """Remove wake phrase from start so agent gets clean input."""
        if not text:
            return text
        t = text.strip().lower()
        if t.startswith(self.wake_word):
            out = text[len(self.wake_word):].strip().lstrip(",").strip()
            return out or text
        for variant in self._WAKE_VARIANTS:
            if t.startswith(variant):
                out = text[len(variant):].strip().lstrip(",").strip()
                return out or text
        if t.startswith("jarvis"):
            out = text[6:].strip().lstrip(",").strip()
            return out or text
        words = t.split()
        if words and self._is_close(words[0], "jarvis", max_distance=2):
            out = text[len(words[0]):].strip().lstrip(",").strip()
            return out or text
        return text

    # ── Common Whisper mishearings ────────────────────────────────────────
    _WAKE_VARIANTS = [
        "hey jarvis", "hey jarves", "hey jarvus", "hey jarvas",
        "hey jelvis", "hey jelby", "hey jervis", "hey jarvy",
        "hey jarv", "hey jarbis", "hey jarbus", "hey jabis",
        "hey jovis", "hey jorvis", "hey jarfis", "hey jarvice",
        "a jarvis", "hay jarvis", "hey javis", "hey jarvi",
        "hey jarby", "hey jarbes", "hey jarbs",
    ]

    # ── Fuzzy wake word matching ──────────────────────────────────────────
    def _matches_wake_word(self, text: str) -> bool:
        """Fuzzy wake word matching — handles STT mishearings."""
        if not text:
            return False
        t = re.sub(r'[^\w\s]', '', text.lower()).strip()
        # Exact substring
        if self.wake_word in t:
            return True
        # Known variants
        for variant in self._WAKE_VARIANTS:
            if variant in t:
                return True
        # Levenshtein on 2-word windows
        words = t.split()
        for i in range(len(words) - 1):
            pair = f"{words[i]} {words[i+1]}"
            if self._is_close(pair, self.wake_word, max_distance=3):
                return True
        # Single word close to "jarvis"
        for word in words:
            if self._is_close(word, "jarvis", max_distance=2):
                return True
        return False

    @staticmethod
    def _is_close(a: str, b: str, max_distance: int = 3) -> bool:
        """Levenshtein distance check."""
        if abs(len(a) - len(b)) > max_distance:
            return False
        m, n = len(a), len(b)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev, dp[0] = dp[0], i
            for j in range(1, n + 1):
                temp = dp[j]
                dp[j] = min(
                    dp[j] + 1,
                    dp[j - 1] + 1,
                    prev + (0 if a[i - 1] == b[j - 1] else 1),
                )
                prev = temp
        return dp[n] <= max_distance

    # ── Start / Stop ──────────────────────────────────────────────────────
    def start(self, callback=None):
        if callback is not None:
            self.on_detected = callback
        if not HAS_AUDIO_TO_TEXT:
            logger.error("[Wake] STT not available. Install faster-whisper: pip install faster-whisper")
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        threading.Thread(target=self._transcribe_loop, daemon=True).start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()

    # ── Audio capture ─────────────────────────────────────────────────────
    def _listen_loop(self):
        """Record audio via sounddevice in a background thread."""
        try:
            import sounddevice as sd
            import numpy as np
            blocksize = 3200  # 0.2s at 16kHz
            device = self.input_device

            def callback(indata, frames, time_info, status):
                if status:
                    return
                chunk = indata[:, 0] if indata.ndim > 1 else indata
                data = chunk.astype(np.int16).tobytes()
                try:
                    self.audio_queue.put_nowait(data)
                except Exception:
                    pass

            with sd.InputStream(
                samplerate=16000, channels=1, dtype="int16",
                blocksize=blocksize, callback=callback, device=device,
            ):
                while self._running:
                    time.sleep(0.1)
        except Exception as e:
            logger.error(f"[Wake] Audio capture error: {e}")

    # ── Transcription + wake detection ────────────────────────────────────
    def _transcribe_loop(self):
        # Allowed short commands that pass validation even with < min_words
        allowed_short = [
            "hey jarvis", "stop", "goodbye", "exit", "cancel",
            "shut down", "go to sleep", "what time",
        ]

        while self._running or not self.audio_queue.empty():
            try:
                # ── If muted (JARVIS speaking), drain audio queue and skip ──
                if self._muted.is_set():
                    try:
                        while not self.audio_queue.empty():
                            self.audio_queue.get_nowait()
                    except queue.Empty:
                        pass
                    time.sleep(0.2)
                    continue

                # ── Collect ~2 seconds of audio ──
                frames = []
                for _ in range(10):
                    try:
                        frames.append(self.audio_queue.get(timeout=0.5))
                    except queue.Empty:
                        break
                if len(frames) < 5:
                    continue

                audio_bytes = b''.join(frames)

                # ── Pre-STT VAD gate: skip if no voice activity ──
                if HAS_VALIDATION:
                    has_speech, reason = has_voice_activity(audio_bytes, sample_rate=16000)
                    if not has_speech:
                        self._chunk_count += 1
                        if self.verbose and (self._chunk_count <= 3 or self._chunk_count % 30 == 0):
                            logger.debug(f"[Wake] Listening... (chunk {self._chunk_count}, {reason})")
                        continue

                # ── Write temp WAV and transcribe ──
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    wf = wave.open(f, 'wb')
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(audio_bytes)
                    wf.close()
                    temp_path = f.name

                transcript = ""
                if HAS_AUDIO_TO_TEXT:
                    try:
                        if not self._stt_loaded:
                            logger.info("[Wake] Loading STT model (first run may take 20-60s)...")
                            self._stt_loaded = True
                        transcript = audio_to_text.transcribe(
                            temp_path,
                            initial_prompt="Hey Jarvis",
                        )
                    except Exception:
                        transcript = ""
                    finally:
                        try:
                            os.unlink(temp_path)
                        except OSError:
                            pass

                self._chunk_count += 1

                # ── Post-STT validation: reject hallucinated noise ──
                if transcript and HAS_VALIDATION:
                    valid, reject_reason = is_valid_transcript(
                        transcript,
                        min_words=2,
                        allowed_short=allowed_short,
                        log_rejections=self.verbose,
                    )
                    if not valid:
                        if self.verbose:
                            logger.debug(f"[Wake] Rejected: {transcript!r} ({reject_reason})")
                        transcript = ""

                # ── Log what we heard ──
                if self.verbose and transcript:
                    logger.debug(f"[Wake] Heard: {transcript[:80]}")
                elif self.verbose and (self._chunk_count <= 3 or self._chunk_count % 30 == 0):
                    logger.debug(f"[Wake] Listening... (chunk {self._chunk_count})")

                # ── Combine with previous chunk (catches "Hey" | "Jarvis, ..." splits) ──
                combined = (
                    f"{self._last_transcript} {transcript}".strip().lower()
                    if self._last_transcript and transcript
                    else (transcript.lower() if transcript else "")
                )
                prev_transcript = self._last_transcript
                self._last_transcript = transcript if transcript else ""
                now = time.time()

                # ── Session timeout ──
                if self._session_active and (now - self._last_speech_time) > self._session_timeout:
                    self._session_active = False
                    logger.info("[Wake] Session expired (silence timeout). Say 'Hey Jarvis' to continue.")

                # ── Wake word detection ──
                wake_detected = combined and self._matches_wake_word(combined)
                in_active_session = self._session_active and transcript and len(transcript.strip()) >= 3

                if wake_detected:
                    if now - self._last_trigger < self.debounce:
                        continue
                    self._last_trigger = now
                    self._session_active = True
                    self._last_speech_time = now
                    phrase = f"{prev_transcript} {transcript}".strip() if prev_transcript else transcript
                    phrase = self._strip_wake_phrase(phrase) or "What can I help you with?"
                    logger.info(f"[Wake] \"{phrase}\"")
                    try:
                        if self.on_detected:
                            self.on_detected(phrase)
                    except Exception as e:
                        logger.error(f"[Wake] Callback error: {e}")

                elif in_active_session:
                    self._last_speech_time = now
                    logger.info(f"[Session] \"{transcript[:80]}\"")
                    try:
                        if self.on_detected:
                            self.on_detected(transcript)
                    except Exception as e:
                        logger.error(f"[Wake] Callback error: {e}")

            except Exception as e:
                logger.error(f"[Wake] Transcribe loop error: {e}")
