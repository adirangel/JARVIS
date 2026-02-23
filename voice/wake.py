import threading
import time
import queue
import wave
import tempfile
import os
from agent.utils import color_print

try:
    import audio_to_text
    HAS_AUDIO_TO_TEXT = True
except ImportError:
    HAS_AUDIO_TO_TEXT = False

class WakeListener:
    def __init__(self, config=None, wake_word="Hey Jarvis", debounce_seconds=1.5, on_detected=None, verbose=False):
        if isinstance(config, dict):
            ww = config.get("wake_word", "Hey Jarvis")
            self.wake_word = (ww if isinstance(ww, str) else "Hey Jarvis").lower()
            self.debounce = config.get("debounce_seconds", 1.5)
            self.verbose = config.get("wake_verbose", False)
            self.api_key = config.get("stt_api_key") or config.get("llm_api_key")
            # Device: support input_device, wake_word.device (config.yaml), or wake_device
            ww_cfg = config.get("wake_word", {}) if isinstance(config.get("wake_word"), dict) else {}
            self.input_device = config.get("input_device") or config.get("wake_device") or ww_cfg.get("device")
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
        self._last_transcript = ""  # For cross-chunk wake detection
        self._session_active = False
        self._last_speech_time = 0.0
        self._session_timeout = float(config.get("session_idle_timeout_seconds", 600)) if isinstance(config, dict) else 600.0
        self.audio_queue = queue.Queue()
    
    def _strip_wake_phrase(self, text: str) -> str:
        """Remove wake phrase from start so agent gets clean input."""
        if not text:
            return text
        t = text.strip().lower()
        if t.startswith(self.wake_word):
            out = text[len(self.wake_word):].strip()
            if out.startswith(","):
                out = out[1:].strip()
            return out or text
        if t.startswith("jarvis"):
            out = text[6:].strip()
            if out.startswith(","):
                out = out[1:].strip()
            return out or text
        return text

    def start(self, callback=None):
        if callback is not None:
            self.on_detected = callback
        if not HAS_AUDIO_TO_TEXT:
            color_print('error', "[Wake] STT not available (audio_to_text). Install faster-whisper: pip install faster-whisper")
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        threading.Thread(target=self._transcribe_loop, daemon=True).start()
    
    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()
    
    def _listen_loop(self):
        """Record audio via sounddevice (same lib as test_mic - ensures same device behavior)."""
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
                samplerate=16000,
                channels=1,
                dtype="int16",
                blocksize=blocksize,
                callback=callback,
                device=device,
            ):
                while self._running:
                    import time
                    time.sleep(0.1)
        except Exception as e:
            color_print('error', f"WakeListener audio error: {e}")
    
    def _transcribe_loop(self):
        while self._running or not self.audio_queue.empty():
            try:
                frames = []
                # collect ~2 seconds of audio
                for _ in range(10):
                    try:
                        frames.append(self.audio_queue.get(timeout=0.5))
                    except queue.Empty:
                        break
                if len(frames) < 5:
                    continue
                # write temp wav
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    wf = wave.open(f, 'wb')
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(b''.join(frames))
                    wf.close()
                    temp_path = f.name
                if HAS_AUDIO_TO_TEXT:
                    try:
                        if not self._stt_loaded:
                            color_print('info', "[Wake] Loading STT model (first run may take 20-60s)...")
                            self._stt_loaded = True
                        transcript = audio_to_text.transcribe(temp_path, self.api_key or "")
                    except Exception:
                        transcript = ""
                    finally:
                        try:
                            os.unlink(temp_path)
                        except:
                            pass
                else:
                    transcript = ""
                self._chunk_count += 1
                if self.verbose:
                    if transcript:
                        color_print('thought', f"[Wake] Heard: {transcript[:80]}")
                    elif self._chunk_count <= 3 or self._chunk_count % 15 == 0:
                        color_print('debug', f"[Wake] Listening... (chunk {self._chunk_count}, no speech yet)")
                # Check current chunk + overlap with previous (catches "Hey" | "Jarvis, ..." split)
                combined = (
                    f"{self._last_transcript} {transcript}".strip().lower()
                    if self._last_transcript and transcript
                    else (transcript.lower() if transcript else "")
                )
                prev_transcript = self._last_transcript
                self._last_transcript = transcript if transcript else ""
                now = time.time()

                # Session timeout: 10 min silence -> require "Hey Jarvis" to re-wake
                if self._session_active and (now - self._last_speech_time) > self._session_timeout:
                    self._session_active = False
                    if self.verbose:
                        color_print('info', "[Wake] Session expired (10 min silence). Say 'Hey Jarvis' to continue.")

                # Decide whether to fire
                wake_detected = combined and self.wake_word in combined  # Only "Hey Jarvis" wakes
                in_active_session = self._session_active and transcript and len(transcript.strip()) >= 3

                if wake_detected:
                    if now - self._last_trigger < self.debounce:
                        if self.verbose:
                            color_print('warn', f"Debounced wake trigger ({(now-self._last_trigger):.2f}s)")
                    else:
                        self._last_trigger = now
                        self._session_active = True
                        self._last_speech_time = now
                        phrase = f"{prev_transcript} {transcript}".strip() if prev_transcript else transcript
                        phrase = self._strip_wake_phrase(phrase) or "What can I help you with?"
                        if self.verbose:
                            color_print('thought', f"Wake: {phrase}")
                        try:
                            if self.on_detected:
                                self.on_detected(phrase)
                        except Exception as e:
                            color_print('error', f"Wake callback error: {e}")
                elif in_active_session:
                    self._last_speech_time = now
                    if self.verbose:
                        color_print('thought', f"[Session] {transcript[:60]}")
                    try:
                        if self.on_detected:
                            self.on_detected(transcript)
                    except Exception as e:
                        color_print('error', f"Wake callback error: {e}")
            except Exception as e:
                color_print('error', f"Transcribe loop error: {e}")
