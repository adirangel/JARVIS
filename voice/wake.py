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
            self.wake_word = config.get("wake_word", "Hey Jarvis").lower()
            self.debounce = config.get("debounce_seconds", 1.5)
            self.verbose = config.get("wake_verbose", False)
            self.api_key = config.get("stt_api_key") or config.get("llm_api_key")
        else:
            self.wake_word = wake_word.lower()
            self.debounce = debounce_seconds
            self.verbose = verbose
            self.api_key = None
        self.on_detected = on_detected
        self._running = False
        self._thread = None
        self._last_trigger = 0
        self.audio_queue = queue.Queue()
    
    def start(self, callback=None):
        if callback is not None:
            self.on_detected = callback
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        threading.Thread(target=self._transcribe_loop, daemon=True).start()
    
    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()
    
    def _listen_loop(self):
        try:
            import pyaudio
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=3200)
            while self._running:
                try:
                    data = stream.read(3200, exception_on_overflow=False)
                    self.audio_queue.put(data)
                except Exception:
                    pass
            stream.stop_stream()
            stream.close()
            p.terminate()
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
                if transcript and self.wake_word in transcript.lower():
                    now = time.time()
                    if now - self._last_trigger < self.debounce:
                        if self.verbose:
                            color_print('warn', f"Debounced wake trigger ({(now-self._last_trigger):.2f}s)")
                        continue
                    self._last_trigger = now
                    if self.verbose:
                        color_print('thought', f"Wake word detected: {transcript}")
                    # fire the callback (may be async or sync)
                    try:
                        if self.on_detected:
                            self.on_detected(transcript)
                    except Exception as e:
                        color_print('error', f"Wake callback error: {e}")
            except Exception as e:
                color_print('error', f"Transcribe loop error: {e}")
