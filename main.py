"""JARVIS - 100% Local AI Assistant.

System tray app. Wake word -> STT -> Agent -> TTS.
Hybrid LLM: DictaLM (conversation) + Qwen3 (tools).
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

# Windows: use UTF-8 so Hebrew/Unicode in agent responses don't trigger charmap errors
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Project root
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_config() -> dict:
    import yaml
    cfg = ROOT / "config.yaml"
    example = ROOT / "config.example.yaml"
    path = cfg if cfg.exists() else example
    config = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    # Patch ddgs to use fixed Chrome (not random) - avoids "Impersonate 'chrome_127' does not exist"
    try:
        from agent.ddgs_patch import apply_ddgs_patch
        apply_ddgs_patch(config)
    except Exception:
        pass
    return config


def create_memory(config: dict) -> "MemoryManager":
    from memory.manager import MemoryManager
    mem_cfg = config.get("memory", {})
    llm_host = config.get("llm", {}).get("host", "http://localhost:11434")
    return MemoryManager(
        db_path=str(ROOT / mem_cfg.get("db_path", "data/jarvis.db")),
        chroma_path=str(ROOT / mem_cfg.get("chroma_path", "data/chroma")),
        embedding_model=mem_cfg.get("embedding_model", "nomic-embed-text"),
        ollama_host=llm_host,
        max_memories=mem_cfg.get("max_memories", 5),
    )


def _display_text(text: str) -> str:
    """Convert Hebrew/RTL text to visual order for correct console display."""
    from voice.tts import _rtl_display
    return _rtl_display(text)


def create_tts(config: dict):
    from voice.tts import create_tts
    v = config.get("voice", {})
    return create_tts(
        engine=v.get("tts_engine", "piper"),
        quality=v.get("tts_quality", "medium"),
        hebrew_voice=v.get("hebrew_voice", "he-IL-AvriNeural"),
        speed=v.get("tts_speed", 1.15),
        force_hebrew_tts=v.get("force_hebrew_tts", False),
        preload=v.get("preload_tts", True),  # Preload Piper - "As you wish, Sir..." within 800ms
    )


def create_stt(config: dict):
    from voice.stt import SpeechToText
    v = config.get("voice", {})
    return SpeechToText(
        model_name=v.get("stt_model", "large-v3-turbo"),
        device=v.get("stt_device", "cuda"),
        language=v.get("stt_language"),
        beam_size=v.get("stt_beam_size", 3),
        compute_type=v.get("stt_compute_type", "int8"),
    )


def _make_streaming_tts_callback(tts, verbose: bool):
    """Stream callback: buffer chunks, queue for immediate TTS on phrase boundaries (sub-800ms first audio)."""
    import queue
    buffer = []
    text_queue = queue.Queue()
    _stop_sentinel = object()

    def _consumer() -> None:
        import sounddevice as sd
        import soundfile as sf
        while True:
            try:
                item = text_queue.get(timeout=0.5)
                if item is _stop_sentinel:
                    break
                if item and str(item).strip():
                    try:
                        wav_path = tts.synthesize(str(item).strip())
                        data, sr = sf.read(wav_path)
                        sd.play(data, sr)
                        sd.wait()
                    except Exception as e:
                        if verbose:
                            print(f"[Stream TTS] {e}", flush=True)
            except queue.Empty:
                continue

    def on_chunk(chunk: str) -> None:
        buffer.append(chunk)
        s = "".join(buffer)
        # Issue 3: Buffer until full sentence to avoid TTS cutoff. Play on sentence end (preferred)
        # or after ~80 chars for long run-ons (avoids cutting "10:58 AM (IST, UTC+2)" mid-time)
        sentence_end = any(s.rstrip().endswith(p) for p in (".", "!", "?", "。", "…", ";"))
        min_chars = 80  # Long enough for time strings; short enough to avoid latency
        if sentence_end or len(s) >= min_chars:
            phrase = "".join(buffer).strip()
            if phrase:
                text_queue.put(phrase)
            buffer.clear()

    def flush() -> str:
        remainder = "".join(buffer).strip()
        buffer.clear()
        return remainder

    def put_remainder(text: str) -> None:
        if text and text.strip():
            # Issue 3: Ensure remainder ends properly (avoids abrupt TTS cutoff)
            from agent.graph import _ensure_complete_sentence
            text = _ensure_complete_sentence(text.strip())
            text_queue.put(text)

    def start_consumer() -> threading.Thread:
        t = threading.Thread(target=_consumer, daemon=True)
        t.start()
        return t

    def stop_consumer() -> None:
        text_queue.put(_stop_sentinel)

    return on_chunk, flush, put_remainder, start_consumer, stop_consumer


def run_jarvis_loop(config: dict, stop_event: threading.Event, console_mode: bool = False, push_to_talk: bool = False):
    """Main loop: wake word -> record -> STT -> agent -> TTS. If console_mode, use text input. If push_to_talk, press Enter to record."""
    from agent.graph import create_jarvis_graph, invoke_jarvis
    from agent.tools import create_tool_router, try_open_browser_from_intent
    from voice.recorder import Recorder

    verbose = console_mode or "--mode" in sys.argv
    if verbose:
        print("Loading memory...", flush=True)
    memory = create_memory(config)
    if verbose:
        print("Loading agent graph (Ollama must be running)...", flush=True)
    graph = create_jarvis_graph(config, memory=memory)
    tool_router = create_tool_router(config)
    if verbose:
        print("Loading TTS...", flush=True)
    tts = create_tts(config)
    if verbose:
        print("Loading STT (first run may download model)...", flush=True)
    stt = create_stt(config)
    stt._ensure_model()

    # Heartbeat: witty voiced summary every 30 min (even when idle)
    def _tts_speak(text: str) -> None:
        try:
            wav_path = tts.synthesize(text)
            import sounddevice as sd
            import soundfile as sf
            data, sr = sf.read(wav_path)
            sd.play(data, sr)
            sd.wait()
        except Exception:
            pass

    def _llm_invoke(prompt: str) -> str:
        try:
            from langchain_ollama import ChatOllama
            from langchain_core.messages import SystemMessage, HumanMessage
            llm_cfg = config.get("llm", {})
            conv_model = llm_cfg.get("conversation_model", "aminadaven/dictalm2.0-instruct:q5_K_M")
            base_url = llm_cfg.get("host", "http://localhost:11434")
            if base_url and not base_url.startswith("http"):
                base_url = f"http://{base_url}"
            llm = ChatOllama(model=conv_model, base_url=base_url, temperature=0.7)
            resp = llm.invoke([
                SystemMessage(content="You are JARVIS. Brief, dry wit. Address user as Sir."),
                HumanMessage(content=prompt),
            ])
            return resp.content if hasattr(resp, "content") else str(resp)
        except Exception:
            return ""

    hb_cfg = config.get("heartbeat", {})
    interval = hb_cfg.get("interval_minutes", 30)
    try:
        from heartbeat import start_heartbeat
        start_heartbeat(memory, _tts_speak, _llm_invoke, interval_minutes=interval)
        if verbose:
            print(f"Heartbeat started (every {interval} min).", flush=True)
    except Exception as e:
        if verbose:
            print(f"Heartbeat not started: {e}", flush=True)

    if console_mode:
        print("Ready. Type your message below.", flush=True)
    elif verbose:
        print("Ready. Listening for 'Hey Jarvis'...", flush=True)

    wake_cfg = config.get("wake_word", {})
    models = wake_cfg.get("models", ["hey_jarvis_v0.1"])
    # Issue 1: Use wake_confidence (0.7-0.85) to reduce false positives; fallback to threshold
    threshold = wake_cfg.get("wake_confidence") or wake_cfg.get("threshold", 0.35)
    mic_device = wake_cfg.get("device")
    noise_gate_rms = wake_cfg.get("noise_gate_rms", 0.005)

    vc = config.get("voice", {})
    recorder = Recorder(
        sample_rate=vc.get("recorder_sample_rate", 16000),
        silence_threshold=vc.get("recorder_silence_threshold", 0.012),
        silence_duration=vc.get("recorder_silence_duration", 2.5),
        device=mic_device,
    )

    try:
        from voice.wake import WakeWordDetector
        wake = WakeWordDetector(
            model_names=models,
            threshold=threshold,
            device=mic_device,
            wake_confidence=wake_cfg.get("wake_confidence"),
            noise_gate_rms=noise_gate_rms,
        )
    except ImportError:
        wake = None

    record_seconds = config.get("voice", {}).get("push_to_talk_seconds", 5)
    wake_cooldown_sec = config.get("wake_word", {}).get("cooldown_seconds", 3)
    _last_wake_time = [0.0]  # mutable for closure
    _processing = [False]  # Only one recording/response at a time (prevents TTS bleed)

    def on_wake():
        import time as _time
        if _processing[0]:
            return  # Already processing - prevents overlapping recordings that capture TTS
        now = _time.monotonic()
        if now - _last_wake_time[0] < wake_cooldown_sec:
            return  # Ignore rapid re-triggers
        _last_wake_time[0] = now
        _processing[0] = True

        def _process():
            try:
                if verbose:
                    print("[Wake word detected] Recording... (speak, I'll wait until you finish)", flush=True)
                audio = recorder.record_until_silence(max_seconds=30)
                if not audio:
                    if verbose:
                        print("[Wake] No audio captured.", flush=True)
                    return
                import numpy as np
                arr = np.frombuffer(audio, dtype=np.int16)
                rms = np.sqrt(np.mean(arr.astype(np.float64) ** 2)) / 32768
                min_rms = config.get("voice", {}).get("min_rms", 0.0020)
                if rms <= min_rms:
                    if verbose:
                        print(f"[Wake] Too quiet (RMS={rms:.4f}), waiting for speech...", flush=True)
                    return
                # Issue 1: Min audio length - ignore very short recordings (noise bursts)
                sample_rate = vc.get("recorder_sample_rate", 16000)
                audio_duration = len(audio) / (sample_rate * 2)
                min_audio_len = vc.get("min_audio_length", 1.0)
                if audio_duration < min_audio_len:
                    if verbose:
                        print(f"[Wake] Audio too short ({audio_duration:.1f}s < {min_audio_len}s), likely noise.", flush=True)
                    return
                import tempfile
                import wave
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    with wave.open(f.name, "wb") as wav:
                        wav.setnchannels(1)
                        wav.setsampwidth(2)
                        wav.setframerate(16000)
                        wav.writeframes(audio)
                    use_vad = vc.get("use_vad", True)
                    text = stt.transcribe(f.name, vad_filter=use_vad)
                # Issue 1: Post-STT validation - discard noise artifacts ("thank you", etc.)
                from voice.validation import is_valid_transcript
                min_words = vc.get("min_transcript_words", 3)
                valid, reason = is_valid_transcript(text, min_words=min_words, log_rejections=verbose)
                if not valid:
                    if verbose:
                        print(f"[Wake] Rejected transcript (false positive: {reason}): {text[:60]!r}", flush=True)
                    return
                if not text.strip():
                    if verbose:
                        print("[Wake] No speech transcribed.", flush=True)
                    return
                if verbose:
                    print(f"[Wake] You said: {_display_text(text)}", flush=True)
                try_open_browser_from_intent(text, tool_router)
                max_words = config.get("voice", {}).get("max_response_words", 50)
                stream_tts = config.get("voice", {}).get("stream_tts", True)
                if stream_tts:
                    on_chunk, flush, put_remainder, start_consumer, stop_consumer = _make_streaming_tts_callback(tts, verbose)
                    consumer_thread = start_consumer()
                    try:
                        response, latency = invoke_jarvis(
                            graph, text,
                            stream_callback=on_chunk,
                            max_words=max_words, config=config, memory=memory,
                        )
                        put_remainder(flush())
                    finally:
                        stop_consumer()
                        consumer_thread.join(timeout=30)
                    # TTS already played via streaming - no duplicate
                else:
                    response, latency = invoke_jarvis(graph, text, stream_callback=None, max_words=max_words, config=config, memory=memory)
                    wav_path = tts.synthesize(response)
                    try:
                        import sounddevice as sd
                        import soundfile as sf
                        data, sr = sf.read(wav_path)
                        sd.play(data, sr)
                        sd.wait()
                    except Exception as e:
                        if verbose:
                            print(f"[Wake] TTS play error: {e}", flush=True)
                show_latency = config.get("show_latency", True)
                if verbose:
                    lat_str = f" [{latency:.1f}s]" if show_latency else ""
                    print(_display_text(response) + lat_str, flush=True)
                memory.save_interaction(text, response)
                import time as _t
                if verbose:
                    _t.sleep(0.5)  # Let audio device settle before next listen
                    print("[Wake] Listening again...", flush=True)
                # Issue 2: Play "Listening, Sir." to indicate ready for follow-up (no wake word needed)
                listening_prompt = vc.get("listening_prompt", "Listening, Sir.")
                if listening_prompt:
                    try:
                        wav_listen = tts.synthesize(listening_prompt)
                        import sounddevice as sd
                        import soundfile as sf
                        data_listen, sr_listen = sf.read(wav_listen)
                        sd.play(data_listen, sr_listen)
                        sd.wait()
                    except Exception:
                        pass
                # Issue 2: Continuous conversation - listen until silence_timeout or explicit end
                silence_timeout = vc.get("silence_timeout") or vc.get("follow_up_seconds", 15)
                _END_COMMANDS = ("goodbye", "good bye", "end", "stop", "that's all", "להתראות", "סיום")
                t_followup_start = _t.monotonic()
                while silence_timeout > 0 and not stop_event.is_set():
                    if verbose:
                        elapsed = _t.monotonic() - t_followup_start
                        print(f"[Wake] Follow-up - speak or stay silent to end... [{elapsed:.1f}s]", flush=True)
                    t_record_start = _t.monotonic()
                    audio2 = recorder.record_until_silence(max_seconds=silence_timeout)
                    if not audio2:
                        break
                    arr2 = np.frombuffer(audio2, dtype=np.int16)
                    rms2 = np.sqrt(np.mean(arr2.astype(np.float64) ** 2)) / 32768
                    if rms2 <= min_rms:
                        if verbose:
                            elapsed = _t.monotonic() - t_record_start
                            print(f"[Wake] Silence - ending conversation. [{elapsed:.1f}s]", flush=True)
                        break
                    # Issue 1: Min audio length for follow-up too
                    audio2_duration = len(audio2) / (sample_rate * 2)
                    if audio2_duration < min_audio_len:
                        if verbose:
                            print(f"[Wake] Follow-up too short ({audio2_duration:.1f}s), skipping.", flush=True)
                        continue
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f2:
                        with wave.open(f2.name, "wb") as wav2:
                            wav2.setnchannels(1)
                            wav2.setsampwidth(2)
                            wav2.setframerate(16000)
                            wav2.writeframes(audio2)
                        text2 = stt.transcribe(f2.name, vad_filter=use_vad)
                    if not text2.strip():
                        if verbose:
                            elapsed = _t.monotonic() - t_record_start
                            print(f"[Wake] No speech - ending conversation. [{elapsed:.1f}s]", flush=True)
                        break
                    # Issue 2: Validate follow-up transcript (avoid false positives)
                    valid2, _ = is_valid_transcript(text2, min_words=min_words, log_rejections=verbose)
                    if not valid2:
                        if verbose:
                            print(f"[Wake] Rejected follow-up (false positive), continuing to listen...", flush=True)
                        continue
                    # Issue 2: Explicit end commands - exit conversation mode
                    t2_lower = text2.strip().lower()
                    if any(cmd in t2_lower for cmd in _END_COMMANDS):
                        if verbose:
                            print("[Wake] User said goodbye - ending conversation.", flush=True)
                        break
                    if verbose:
                        print(f"[Wake] You said: {_display_text(text2)}", flush=True)
                    try_open_browser_from_intent(text2, tool_router)
                    response2, latency2 = invoke_jarvis(graph, text2, stream_callback=None, max_words=max_words, config=config, memory=memory)
                    show_latency = config.get("show_latency", True)
                    if verbose:
                        lat_str = f" [{latency2:.1f}s]" if show_latency else ""
                        print(_display_text(response2) + lat_str, flush=True)
                    memory.save_interaction(text2, response2)
                    wav_path2 = tts.synthesize(response2)
                    try:
                        import sounddevice as sd
                        import soundfile as sf
                        data2, sr2 = sf.read(wav_path2)
                        sd.play(data2, sr2)
                        sd.wait()
                    except Exception:
                        pass
                    if verbose:
                        _t.sleep(0.5)
                        print("[Wake] Listening again...", flush=True)
                    # Issue 2: Play listening prompt after each response in continuous mode
                    if listening_prompt:
                        try:
                            wav2 = tts.synthesize(listening_prompt)
                            import sounddevice as sd
                            import soundfile as sf
                            d2, sr2 = sf.read(wav2)
                            sd.play(d2, sr2)
                            sd.wait()
                        except Exception:
                            pass
            except Exception as e:
                print(f"[Wake] Error: {e}", flush=True)
            finally:
                _processing[0] = False
        threading.Thread(target=_process, daemon=True).start()

    if push_to_talk and not console_mode:
        # Push-to-talk: press Enter, speak for N seconds (fixed duration = more reliable)
        while not stop_event.is_set():
            try:
                input(f"Press Enter to speak ({record_seconds}s recording)... ")
                print("Listening...", flush=True)
                try:
                    audio = recorder.record_fixed(seconds=record_seconds)
                    if not audio:
                        print("No audio captured.", flush=True)
                        continue
                    import numpy as np
                    arr = np.frombuffer(audio, dtype=np.int16)
                    rms = np.sqrt(np.mean(arr.astype(np.float64) ** 2)) / 32768
                    min_rms = config.get("voice", {}).get("min_rms", 0.0020)
                    print(f"Recorded: RMS={rms:.4f} ({'OK' if rms > min_rms else 'TOO QUIET'})", flush=True)
                    if rms <= min_rms:
                        print("Too quiet. Speak louder and try again.", flush=True)
                        continue
                    debug_path = ROOT / "last_recording.wav"
                    import wave
                    with wave.open(str(debug_path), "wb") as wav:
                        wav.setnchannels(1)
                        wav.setsampwidth(2)
                        wav.setframerate(16000)
                        wav.writeframes(audio)
                    # Transcribe
                    text = stt.transcribe(str(debug_path))
                    if not text.strip():
                        print("Could not transcribe. Check last_recording.wav", flush=True)
                        continue
                    print(f"You said: {_display_text(text)}", flush=True)
                    try_open_browser_from_intent(text, tool_router)
                    max_words = config.get("voice", {}).get("max_response_words", 50)
                    print("JARVIS: ", end="", flush=True)
                    response, latency = invoke_jarvis(graph, text, stream_callback=None, max_words=max_words, config=config, memory=memory)
                    show_latency = config.get("show_latency", True)
                    lat_str = f" [{latency:.1f}s]" if show_latency else ""
                    print(_display_text(response) + lat_str, flush=True)
                    memory.save_interaction(text, response)
                    print()  # blank line before next prompt
                    wav_path = tts.synthesize(response)
                    try:
                        import sounddevice as sd
                        import soundfile as sf
                        data, sr = sf.read(wav_path)
                        sd.play(data, sr)
                        sd.wait()
                    except Exception:
                        pass
                except Exception as e:
                    print(f"Error: {e}", flush=True)
            except (EOFError, KeyboardInterrupt):
                break
    elif wake and not console_mode:
        # Verbose in voice mode so user sees wake detection and processing
        verbose = config.get("general", {}).get("debug", False) or config.get("debug", False) or "--mode" in sys.argv
        show_scores = config.get("wake_word", {}).get("show_scores", False) or "--debug-wake" in sys.argv
        if verbose:
            print("Listening for 'Hey Jarvis' (say it now)...", flush=True)
        if show_scores:
            print("Debug: showing wake scores every ~4s. Say 'Hey Jarvis' to test.", flush=True)
        try:
            wake.listen(on_detected=on_wake, verbose=verbose, show_scores=show_scores)
        except Exception as e:
            print(f"[Wake] Listener error: {e}", flush=True)
    else:
        # Fallback: no wake word, just run a simple input loop for testing
        max_words = config.get("voice", {}).get("max_response_words", 50)
        while not stop_event.is_set():
            try:
                text = input("You: ").strip()
                if not text:
                    continue
                try_open_browser_from_intent(text, tool_router)
                print("JARVIS: ", end="", flush=True)
                response, latency = invoke_jarvis(graph, text, stream_callback=None, max_words=max_words, config=config, memory=memory)
                show_latency = config.get("show_latency", True)
                lat_str = f" [{latency:.1f}s]" if show_latency else ""
                print(_display_text(response) + lat_str, flush=True)
                memory.save_interaction(text, response)
                wav_path = tts.synthesize(response)
                try:
                    import sounddevice as sd
                    import soundfile as sf
                    data, sr = sf.read(wav_path)
                    sd.play(data, sr)
                    sd.wait()
                except Exception:
                    pass
            except (EOFError, KeyboardInterrupt):
                break


def create_tray_icon(config: dict):
    """System tray with pystray."""
    import pystray
    from PIL import Image

    # Create a simple icon
    size = 64
    img = Image.new("RGB", (size, size), color=(30, 60, 120))
    icon = pystray.Icon("jarvis", img, "JARVIS")

    stop_event = threading.Event()
    loop_thread = None

    def on_start(icon, item):
        nonlocal loop_thread
        if loop_thread and loop_thread.is_alive():
            return
        stop_event.clear()
        loop_thread = threading.Thread(target=run_jarvis_loop, args=(config, stop_event), kwargs={"console_mode": False}, daemon=True)
        loop_thread.start()

    def on_stop(icon, item):
        stop_event.set()

    def on_quit(icon, item):
        stop_event.set()
        icon.stop()

    icon.menu = pystray.Menu(
        pystray.MenuItem("Start", on_start),
        pystray.MenuItem("Stop", on_stop),
        pystray.MenuItem("Quit", on_quit),
    )
    return icon


def main():
    config = load_config()

    # Console mode: text input, no tray
    if "--console" in sys.argv or "-c" in sys.argv:
        print("JARVIS console mode. Type your message and press Enter. Ctrl+C to quit.")
        stop_event = threading.Event()
        run_jarvis_loop(config, stop_event, console_mode=True)
        return

    # Voice mode: wake word + voice directly, no tray (runs in foreground)
    if "--mode" in sys.argv:
        idx = sys.argv.index("--mode")
        mode = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else ""
        if mode.lower() == "voice":
            push_to_talk = "--push-to-talk" in sys.argv or "-p" in sys.argv
            if push_to_talk:
                print("JARVIS push-to-talk. Press Enter, speak, press Enter again. Ctrl+C to quit.")
            else:
                print("JARVIS voice mode. Say 'Hey Jarvis' then speak. Ctrl+C to quit.")
            stop_event = threading.Event()
            run_jarvis_loop(config, stop_event, console_mode=False, push_to_talk=push_to_talk)
            return

    # System tray mode (default)
    print("JARVIS tray mode. Look for the icon in your system tray (bottom-right).")
    print("Right-click the icon -> Start to begin. The terminal will stay open.", flush=True)
    icon = create_tray_icon(config)
    icon.run()


if __name__ == "__main__":
    main()
