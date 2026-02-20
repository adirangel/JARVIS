"""Test wake word detection. Prints scores in real time so you can verify "Hey Jarvis" triggers.

Run: python test_wake.py
Say "Hey Jarvis" - you should see scores spike above threshold.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import yaml

def main():
    cfg_path = ROOT / "config.yaml"
    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    wake_cfg = config.get("wake_word", {})
    models = wake_cfg.get("models", ["hey_jarvis_v0.1"])
    threshold = wake_cfg.get("threshold", 0.35)

    from voice.wake import WakeWordDetector
    wake = WakeWordDetector(model_names=models, threshold=threshold, device=wake_cfg.get("device"))

    import sounddevice as sd
    import numpy as np

    print(f"Listening for 'Hey Jarvis' (threshold={threshold}). Say it now...\n")

    def callback(indata, frames, time_info, status):
        if status:
            return
        chunk = indata[:, 0].tobytes() if indata.ndim > 1 else indata.tobytes()
        pred = wake.predict(chunk)
        for name, score in pred.items():
            s = float(score) if not hasattr(score, "__iter__") else (max(score) if score else 0)
            triggered = "*** DETECTED ***" if s >= threshold else ""
            print(f"\r{name}: {s:.3f}  {triggered}", end="", flush=True)

    with sd.InputStream(samplerate=16000, channels=1, dtype="int16", blocksize=1280, callback=callback, device=wake_cfg.get("device")):
        try:
            while True:
                import time
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nDone.")

if __name__ == "__main__":
    main()
