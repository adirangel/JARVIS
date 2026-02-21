"""Verify voice pipeline and upgraded loop behavior.

Run:
    python verify_voice.py

Checks:
1. Mic capture (RMS)
2. STT + language detection
3. Active-session state machine (3-turn continuity)
4. Noise filtering expectations
5. Long Hebrew TTS synthesis (no cut)
6. Agent response
"""

import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    import wave

    import numpy as np
    import yaml

    print("=== JARVIS Voice Verify (Continuous Session Edition) ===\n")
    cfg_path = ROOT / "config.yaml"
    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    voice_cfg = config.get("voice", {})
    wake_cfg = config.get("wake_word", {})

    # 1) Mic
    print("1) Testing microphone (5s)...")
    from voice.recorder import Recorder

    recorder = Recorder(
        sample_rate=voice_cfg.get("recorder_sample_rate", 16000),
        silence_threshold=voice_cfg.get("recorder_silence_threshold", 0.012),
        silence_duration=voice_cfg.get("recorder_silence_duration", 2.5),
        device=wake_cfg.get("device"),
    )
    audio = recorder.record_fixed(5.0)
    rms = np.sqrt(np.mean(np.frombuffer(audio, dtype=np.int16).astype(np.float64) ** 2)) / 32768
    print(f"   RMS={rms:.4f} {'OK' if rms > 0.001 else 'TOO QUIET'}\n")

    # 2) STT
    print("2) Testing STT + language detection...")
    from voice.stt import SpeechToText

    stt = SpeechToText(
        model_name=voice_cfg.get("stt_model", "large-v3-turbo"),
        device=voice_cfg.get("stt_device", "cuda"),
        language=voice_cfg.get("stt_language"),
        beam_size=voice_cfg.get("stt_beam_size", 3),
        compute_type=voice_cfg.get("stt_compute_type", "int8"),
    )
    stt._ensure_model()
    rec_path = ROOT / "verify_recording.wav"
    with wave.open(str(rec_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(voice_cfg.get("recorder_sample_rate", 16000))
        wav.writeframes(audio)
    details = stt.transcribe_detailed(str(rec_path), vad_filter=voice_cfg.get("use_vad", True))
    text = details.get("text", "").strip()
    lang = details.get("language", "unknown")
    print(f"   Language={lang} | Text='{text[:80]}'\n")

    # 3) Session state machine
    print("3) Testing active-session continuity logic (3 turns)...")
    from voice.session import VoiceSession

    sess = VoiceSession(silence_timeout=voice_cfg.get("silence_timeout", 15))
    sess.activate()
    ok = sess.is_active()
    for _ in range(3):
        sess.touch()
        ok = ok and sess.is_active() and not sess.timed_out()
    print(f"   Continuous 3-turn session: {'OK' if ok else 'FAILED'}\n")

    # 4) Noise filtering expectations
    print("4) Testing noise filtering expectations...")
    from voice.validation import is_valid_transcript

    v1, r1 = is_valid_transcript("thank you", min_words=voice_cfg.get("min_transcript_words", 3))
    v2, r2 = is_valid_transcript("background noise", min_words=voice_cfg.get("min_transcript_words", 3))
    print(f"   'thank you' accepted={v1} reason={r1}")
    print(f"   'background noise' accepted={v2} reason={r2}\n")

    # 5) Long Hebrew TTS (no cut)
    print("5) Testing long Hebrew TTS (chunked, no truncation)...")
    from voice.tts import create_tts

    tts = create_tts(
        engine=voice_cfg.get("tts_engine", "piper"),
        quality=voice_cfg.get("tts_quality", "medium"),
        hebrew_voice=voice_cfg.get("hebrew_voice"),
        speed=voice_cfg.get("tts_speed", 1.0),
        force_hebrew_tts=voice_cfg.get("force_hebrew_tts", False),
        preload=False,
        hebrew_model_repo=voice_cfg.get("hebrew_model_repo"),
        hebrew_model_path=voice_cfg.get("hebrew_model_path"),
        hebrew_model_config=voice_cfg.get("hebrew_model_config"),
        allow_remote_hebrew_fallback=voice_cfg.get("allow_remote_hebrew_fallback", False),
    )
    long_he = (
        "שלום Sir, מערכת הקול פעילה. "
        "אני ממשיך להאזין ברצף גם לשאלות המשך, "
        "ומסיים את ההפעלה רק לפי פקודת סיום או timeout."
    )
    wav_path = tts.synthesize(long_he, language_hint="he")
    try:
        import soundfile as sf

        info = sf.info(wav_path)
        duration = info.frames / float(info.samplerate) if info.samplerate else 0.0
    except Exception:
        duration = 0.0
    print(f"   Synthesized long Hebrew to: {wav_path}")
    print(f"   Duration: {duration:.2f}s {'OK' if duration > 1.0 else 'CHECK'}\n")

    # 6) Agent response
    print("6) Testing agent...")
    try:
        from agent.graph import create_jarvis_graph, invoke_jarvis

        graph = create_jarvis_graph(config, checkpointer_path="data/verify_cp")
        reply, lat = invoke_jarvis(graph, "Say one short sentence confirming voice loop readiness.", max_words=0, config=config)
        print(f"   Agent reply ({lat:.2f}s): {reply[:120]}\n")
    except Exception as e:
        print(f"   Agent test failed: {e}\n")

    print("=== Verify complete ===")
    print("Recommended manual checks:")
    print("1) Noisy room false positives: python test_wake.py --noise-file samples/noise.wav")
    print("2) Three-turn conversation: python main.py --mode voice")
    print("3) Long Hebrew response: ask for a detailed Hebrew summary in voice mode")


if __name__ == "__main__":
    main()
