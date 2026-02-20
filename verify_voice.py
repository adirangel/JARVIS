"""Verify full voice pipeline: mic -> STT -> agent -> TTS.

Run: python verify_voice.py

Tests:
1. Microphone recording
2. STT (faster-whisper) - GPU or CPU
3. Agent (Ollama must be running)
4. TTS (Piper)
"""

import sys
from pathlib import Path

# Windows: use UTF-8 so Hebrew/Unicode in agent responses don't trigger charmap errors
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    import yaml

    print("=== JARVIS Voice Pipeline Verify ===\n")

    # Load config
    cfg_path = ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    # 1. Mic
    print("1. Testing microphone (5s)...")
    from voice.recorder import Recorder
    wake_cfg = config.get("wake_word", {})
    recorder = Recorder(
        sample_rate=16000,
        device=wake_cfg.get("device"),
    )
    audio = recorder.record_fixed(5.0)
    import numpy as np
    rms = np.sqrt(np.mean(np.frombuffer(audio, dtype=np.int16).astype(np.float64) ** 2)) / 32768
    print(f"   RMS={rms:.4f} {'OK' if rms > 0.001 else 'TOO QUIET'}\n")

    # 2. STT
    print("2. Testing STT (faster-whisper)...")
    from voice.stt import SpeechToText
    voice_cfg = config.get("voice", {})
    stt = SpeechToText(
        model_name=voice_cfg.get("stt_model", "large-v3-turbo"),
        device=voice_cfg.get("stt_device", "cuda"),
    )
    stt._ensure_model()
    import wave
    verify_path = ROOT / "verify_recording.wav"
    with wave.open(str(verify_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(audio)
    text = stt.transcribe(str(verify_path))
    print(f"   Transcribed: '{text[:80]}...' " if len(text) > 80 else f"   Transcribed: '{text}'")
    print()

    # 3. Agent
    print("3. Testing agent (Ollama)...")
    try:
        from agent.graph import create_jarvis_graph, invoke_jarvis
        graph = create_jarvis_graph(config, checkpointer_path="data/verify_cp")
        reply = invoke_jarvis(graph, "Say 'Verification complete' in one short sentence.")
        print(f"   Agent: {reply[:100]}...\n")
    except Exception as e:
        print(f"   FAILED: {e}\n")
        return

    # 4. TTS
    print("4. Testing TTS (Piper)...")
    from voice.tts import create_tts
    tts = create_tts(engine="piper", quality="medium")
    wav_path = tts.synthesize("Verification complete, Sir.")
    print(f"   Synthesized to {wav_path}\n")

    print("=== All voice components OK ===")
    print("Run: python main.py --mode voice")
    print("Say 'Hey Jarvis' then speak.")


if __name__ == "__main__":
    main()
