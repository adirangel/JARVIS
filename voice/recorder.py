"""Audio recorder - record until silence."""

from __future__ import annotations

import queue
import time
from typing import Optional

try:
    import numpy as np
    import sounddevice as sd
    RECORDER_AVAILABLE = True
except ImportError:
    RECORDER_AVAILABLE = False


class Recorder:
    """Record audio until silence detected."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        silence_threshold: float = 0.01,
        silence_duration: float = 1.5,
        device: Optional[int] = None,
    ):
        if not RECORDER_AVAILABLE:
            raise ImportError("numpy and sounddevice required. pip install numpy sounddevice")
        self._sample_rate = sample_rate
        self._channels = channels
        self._silence_threshold = silence_threshold
        self._silence_duration = silence_duration
        self._device = device
        self._q: Optional[queue.Queue] = None

    def record_until_silence(
        self,
        silence_duration: Optional[float] = None,
        max_seconds: float = 30,
    ) -> bytes:
        """Record until silence or max_seconds."""
        duration = silence_duration or self._silence_duration
        self._q = queue.Queue()
        chunks = []
        silent_frames = 0
        frames_per_chunk = 512
        silence_frames_needed = int(duration * self._sample_rate / frames_per_chunk)
        total_frames = 0
        max_frames = int(max_seconds * self._sample_rate / frames_per_chunk)

        def callback(indata, frames, time_info, status):
            self._q.put(indata.copy())

        kwargs = dict(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="int16",
            blocksize=frames_per_chunk,
            callback=callback,
        )
        if self._device is not None:
            kwargs["device"] = self._device
        with sd.InputStream(**kwargs):
            while total_frames < max_frames:
                try:
                    chunk = self._q.get(timeout=0.5)
                    chunks.append(chunk)
                    total_frames += 1
                    rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2)) / 32768
                    if rms < self._silence_threshold:
                        silent_frames += 1
                        if silent_frames >= silence_frames_needed and len(chunks) > 10:
                            break
                    else:
                        silent_frames = 0
                except queue.Empty:
                    if chunks:
                        break

        if not chunks:
            return b""
        return np.concatenate(chunks).tobytes()

    def record_fixed(self, seconds: float = 5.0) -> bytes:
        """Record for a fixed duration. More reliable when silence detection fails."""
        if not RECORDER_AVAILABLE:
            raise ImportError("numpy and sounddevice required")
        sample_rate = self._sample_rate
        frames = int(seconds * sample_rate)
        kwargs = dict(samplerate=sample_rate, channels=1, dtype="int16")
        if self._device is not None:
            kwargs["device"] = self._device
        recording = sd.rec(frames, **kwargs)
        sd.wait()
        return recording.tobytes()
