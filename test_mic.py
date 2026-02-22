"""Test microphone - run this to verify JARVIS can hear you.

Usage: python test_mic.py [device_id]

If no device_id: lists devices and records from default for 5 seconds.
If device_id: records from that device (e.g. python test_mic.py 1)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

def main():
    import sounddevice as sd
    import numpy as np

    if len(sys.argv) > 1:
        try:
            device = int(sys.argv[1])
        except ValueError:
            device = None
    else:
        device = None

    print("=== Audio devices ===")
    print(sd.query_devices())
    print()

    if device is not None:
        print(f"Using device {device}")
    else:
        default_in = sd.query_devices(kind="input")
        print(f"Default input: {default_in['name']}")
        print()

    print("Recording for 5 seconds... SPEAK NOW!")
    duration = 5
    sample_rate = 16000
    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
        device=device,
    )
    sd.wait()

    rms = np.sqrt(np.mean(recording.astype(np.float64) ** 2)) / 32768
    max_val = np.abs(recording).max() / 32768
    print(f"\nDone. RMS level: {rms:.4f} (0.01+ = good, <0.001 = too quiet)")
    print(f"Peak level: {max_val:.4f}")

    if rms < 0.001:
        print("\n*** MICROPHONE TOO QUIET ***")
        print("Check: 1) Correct mic selected 2) Mic volume in Windows 3) Speak closer")
        print("Try: python test_mic.py 1  (or 2, 3... to test other devices)")
    else:
        out_path = ROOT / "test_recording.wav"
        import wave
        with wave.open(str(out_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(recording.tobytes())
        print(f"\nSaved to {out_path} - play it to verify.")
        print("If it sounds good, the issue may be in the recorder's silence detection.")

if __name__ == "__main__":
    main()
