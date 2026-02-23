"""Wake-word tester with optional offline noise-file false-positive check.

Examples:
    python test_wake.py
    python test_wake.py --noise-file samples/noise.wav
"""

import argparse
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import yaml


def _max_score(pred):
    m = 0.0
    for score in pred.values():
        s = float(score) if not hasattr(score, "__iter__") else (max(score) if score else 0.0)
        if s > m:
            m = s
    return m


def _run_noise_file(wake, threshold: float, noise_file: str) -> None:
    p = Path(noise_file)
    if not p.exists():
        print(f"Noise file not found: {p}")
        return
    with wave.open(str(p), "rb") as wav:
        sr = wav.getframerate()
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        if sr != 16000 or channels != 1 or width != 2:
            print("Noise file must be mono 16kHz int16 WAV for direct openwakeword testing.")
            return
        chunk_frames = 1280
        total = 0
        triggers = 0
        max_seen = 0.0
        while True:
            frames = wav.readframes(chunk_frames)
            if not frames:
                break
            pred = wake.predict(frames)
            s = _max_score(pred)
            max_seen = max(max_seen, s)
            if s >= threshold:
                triggers += 1
            total += 1
    ratio = (triggers / total) if total else 0.0
    print(f"Noise-file frames={total} | triggers={triggers} | false-positive ratio={ratio:.4f} | max_score={max_seen:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise-file", type=str, default="")
    args = parser.parse_args()

    cfg_path = ROOT / "config.yaml"
    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    wake_cfg = config.get("wake_word", {})
    models = wake_cfg.get("models", ["hey_jarvis_v0.1"])
    threshold = wake_cfg.get("wake_confidence", wake_cfg.get("threshold", 0.75))

    from voice.wake import WakeWordDetector

    wake = WakeWordDetector(
        model_names=models,
        threshold=threshold,
        device=wake_cfg.get("device"),
        wake_confidence=wake_cfg.get("wake_confidence"),
        noise_gate_rms=wake_cfg.get("noise_gate_rms", 0.005),
    )

    if args.noise_file:
        print(f"Running noise-file false-positive test (threshold={threshold})...")
        _run_noise_file(wake, threshold, args.noise_file)
        return

    import sounddevice as sd

    print(f"Listening for wake word (threshold={threshold}). Say 'Hey Jarvis'...\n")

    def callback(indata, frames, time_info, status):
        if status:
            return
        chunk = indata[:, 0].tobytes() if indata.ndim > 1 else indata.tobytes()
        pred = wake.predict(chunk)
        s = _max_score(pred)
        triggered = "*** DETECTED ***" if s >= threshold else ""
        print(f"\rscore={s:.3f}  {triggered}", end="", flush=True)

    with sd.InputStream(
        samplerate=16000,
        channels=1,
        dtype="int16",
        blocksize=1280,
        callback=callback,
        device=wake_cfg.get("device"),
    ):
        try:
            while True:
                import time

                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nDone.")


if __name__ == "__main__":
    main()
