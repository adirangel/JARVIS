"""Wake word detection using openwakeword.

"Hey Jarvis" (English) - hey_jarvis_v0.1
"""

from __future__ import annotations

import queue
import threading
from typing import Callable, Optional

try:
    import openwakeword
    from openwakeword.model import Model
    OPENWAKEWORD_AVAILABLE = True
except ImportError:
    OPENWAKEWORD_AVAILABLE = False
    Model = None


class WakeWordDetector:
    """Detect 'Hey Jarvis' via openwakeword.

    Issue 1 fix: Higher confidence threshold (0.7-0.85) + noise gate to reduce
    false positives from background noise.
    """

    def __init__(
        self,
        model_names: Optional[list[str]] = None,
        threshold: float = 0.75,
        device: Optional[int] = None,
        wake_confidence: Optional[float] = None,
        noise_gate_rms: Optional[float] = None,
    ):
        if not OPENWAKEWORD_AVAILABLE:
            raise ImportError("openwakeword required. pip install openwakeword")
        self._model_names = model_names or ["hey_jarvis_v0.1"]
        # Prefer wake_confidence (0.7-0.85) over legacy threshold for fewer false positives
        base = wake_confidence if wake_confidence is not None else threshold
        self._threshold = max(0.0, min(float(base), 1.0))
        self._device = device
        self._model = None
        # Noise gate: skip prediction if chunk RMS below this (reduces false triggers)
        self._noise_gate_rms = noise_gate_rms if noise_gate_rms is not None else 0.0

    def _ensure_model(self) -> None:
        if self._model is None:
            try:
                openwakeword.utils.download_models()
            except Exception:
                pass
            self._model = Model(
                wakeword_models=self._model_names,
                inference_framework="onnx",
            )

    def _chunk_has_speech(self, audio_chunk: bytes) -> bool:
        """Noise gate: return False if chunk is too quiet (likely background hiss)."""
        if self._noise_gate_rms <= 0:
            return True
        import numpy as np
        arr = np.frombuffer(audio_chunk, dtype=np.int16)
        rms = np.sqrt(np.mean(arr.astype(np.float64) ** 2)) / 32768
        return rms >= self._noise_gate_rms

    def predict(self, audio_chunk: bytes) -> dict:
        """Return prediction scores for each wake word."""
        self._ensure_model()
        import numpy as np
        arr = np.frombuffer(audio_chunk, dtype=np.int16)
        return self._model.predict(arr)

    def detect(self, audio_chunk: bytes) -> bool:
        """True if wake word detected above threshold."""
        if not self._chunk_has_speech(audio_chunk):
            return False
        preds = self.predict(audio_chunk)
        for scores in preds.values():
            if hasattr(scores, "__iter__") and scores:
                s = max(scores) if isinstance(scores, (list, tuple)) else scores[-1]
                if s >= self._threshold:
                    return True
            elif isinstance(scores, (int, float)) and scores >= self._threshold:
                return True
        return False

    def listen(
        self,
        on_detected: Callable[[], None],
        sample_rate: int = 16000,
        chunk_frames: int = 1280,
        verbose: bool = False,
        show_scores: bool = False,
    ) -> None:
        """Listen for wake word. chunk_frames=1280 (80ms) recommended by openwakeword.
        Uses a queue so the audio callback stays non-blocking (sounddevice requirement).
        show_scores: print max score every ~4s for debugging.
        """
        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError("sounddevice required. pip install sounddevice")

        self._ensure_model()
        chunk_queue: queue.Queue[bytes] = queue.Queue(maxsize=64)

        def callback(indata, frames, time_info, status):
            if status:
                return
            mono = indata[:, 0] if indata.ndim > 1 else indata
            chunk = mono.astype("int16").tobytes()
            try:
                chunk_queue.put_nowait(chunk)
            except queue.Full:
                pass

        def worker():
            chunk_count = 0
            max_score = 0.0
            while True:
                try:
                    chunk = chunk_queue.get(timeout=0.5)
                    # Noise gate: skip prediction on very quiet chunks (reduces false positives)
                    if not self._chunk_has_speech(chunk):
                        continue
                    preds = self.predict(chunk)
                    for scores in preds.values():
                        s = float(scores) if not hasattr(scores, "__iter__") else (max(scores) if scores else 0)
                        max_score = max(max_score, s)
                        if s >= self._threshold:
                            if verbose:
                                print("[Wake word detected]", flush=True)
                            on_detected()
                            max_score = 0.0
                    chunk_count += 1
                    if show_scores and chunk_count >= 50:
                        chunk_count = 0
                        print(f"[Wake] score={max_score:.3f} (threshold={self._threshold})", flush=True)
                        max_score = 0.0
                except queue.Empty:
                    continue

        worker_thread = threading.Thread(target=worker, daemon=True)
        worker_thread.start()

        kwargs = dict(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            blocksize=chunk_frames,
            callback=callback,
        )
        if self._device is not None:
            kwargs["device"] = self._device

        with sd.InputStream(**kwargs):
            import time
            while True:
                time.sleep(0.1)
