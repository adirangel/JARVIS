"""JARVIS - 100% Local AI Assistant.

System tray app. Wake word -> active voice session -> STT -> agent -> TTS.
Single model: qwen3:4b for ALL tasks (conversation, planning, reflection, tools, self-evolution).
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

# Windows: use UTF-8 so Hebrew/Unicode in agent responses don't trigger charmap errors.
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
    # Patch ddgs to use fixed Chrome (not random).
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
        chroma_cache_recent=mem_cfg.get("chroma_cache_recent", True),
        chroma_cache_size=mem_cfg.get("chroma_cache_size", 50),
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
        speed=v.get("tts_speed", 1.15),
        preload=v.get("preload_tts", True),
    )


def create_stt(config: dict):
    from voice.stt import SpeechToText

    v = config.get("voice", {})
    lang = v.get("stt_language")
    if isinstance(lang, str) and lang.strip().lower() in ("", "auto", "detect", "automatic"):
        lang = None
    return SpeechToText(
        model_name=v.get("stt_model", "large-v3-turbo"),
        device=v.get("stt_device", "cuda"),
        language=lang,
        beam_size=v.get("stt_beam_size", 3),
        compute_type=v.get("stt_compute_type", "int8"),
    )


def _play_ready_beep(volume: float = 0.08, seconds: float = 0.08, frequency_hz: float = 920.0) -> None:
    """Soft readiness beep for active-session mode."""
    try:
        import numpy as np
        import sounddevice as sd

        sample_rate = 22050
        t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
        tone = (np.sin(2 * np.pi * frequency_hz * t) * volume).astype(np.float32)
        sd.play(tone, sample_rate)
        sd.wait()
    except Exception:
        pass


def _make_streaming_tts_callback(tts, play_audio_file, verbose: bool):
    """Stream callback: queue sentence-complete chunks for TTS playback."""
    import queue
    import re

    buffer = []
    text_queue = queue.Queue()
    _stop_sentinel = object()
    sentence_boundary = re.compile(r"[.!?\u2026;:\u05C3]\s*$")

    def _consumer() -> None:
        while True:
            try:
                item = text_queue.get(timeout=0.5)
                if item is _stop_sentinel:
                    break
                if item and str(item).strip():
                    try:
                        wav_path = tts.synthesize(str(item).strip())
                        play_audio_file(wav_path)
                    except Exception as e:
                        if verbose:
                            print(f"[Stream TTS] {e}", flush=True)
            except queue.Empty:
                continue

    def on_chunk(chunk: str) -> None:
        buffer.append(chunk)
        s = "".join(buffer)
        # Buffer until sentence punctuation to avoid truncation in playback.
        if sentence_boundary.search(s.rstrip()):
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
    """Main loop: wake word -> active session -> STT -> agent -> TTS."""
    import tempfile
    import wave

    import numpy as np

    from agent.graph import create_jarvis_graph, invoke_jarvis
    from agent.tools import create_tool_router, try_open_browser_from_intent
    from voice.recorder import Recorder
    from voice.session import DEFAULT_END_COMMANDS, VoiceSession
    from voice.validation import has_voice_activity, is_valid_transcript

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

    wake_cfg = config.get("wake_word", {})
    vc = config.get("voice", {})
    sample_rate = int(vc.get("recorder_sample_rate", 16000))
    min_rms = float(vc.get("min_rms", 0.001))
    min_audio_len = float(vc.get("min_audio_length", 1.0))
    min_words = int(vc.get("min_transcript_words", 3))
    use_vad = bool(vc.get("use_vad", True))
    speech_start_timeout = float(vc.get("speech_start_timeout", 3.5))
    max_record_seconds = float(vc.get("max_record_seconds", 30))
    max_words = vc.get("max_response_words", 0) or 0
    silence_timeout = float(vc.get("silence_timeout", 15))
    thinking_prompt_each_turn = bool(vc.get("thinking_prompt_each_turn", False))

    end_commands_cfg = vc.get("session_end_commands", DEFAULT_END_COMMANDS)
    if isinstance(end_commands_cfg, str):
        end_commands_cfg = [p.strip() for p in end_commands_cfg.split(",") if p.strip()]
    session = VoiceSession(
        silence_timeout=silence_timeout,
        end_commands=tuple(end_commands_cfg) if end_commands_cfg else tuple(DEFAULT_END_COMMANDS),
    )

    audio_lock = threading.Lock()

    def _play_audio_file(wav_path: str) -> None:
        import sounddevice as sd
        import soundfile as sf

        with audio_lock:
            data, sr = sf.read(wav_path)
            sd.play(data, sr)
            sd.wait()

    def _speak(text: str, skip_if_active: bool = False) -> None:
        if not text or not text.strip():
            return
        if skip_if_active and session.is_active():
            return
        try:
            wav_path = tts.synthesize(text.strip())
            _play_audio_file(wav_path)
        except Exception as e:
            if verbose:
                print(f"[TTS] {e}", flush=True)

    # Heartbeat: never interrupt active sessions.
    def _tts_speak(text: str) -> None:
        if session.is_active():
            if verbose:
                print("[Heartbeat] Deferred during active session.", flush=True)
            return
        _speak(text, skip_if_active=True)

    def _llm_invoke(prompt: str) -> str:
        try:
            from langchain_ollama import ChatOllama
            from langchain_core.messages import HumanMessage, SystemMessage

            llm_cfg = config.get("llm", {})
            model = llm_cfg.get("model") or llm_cfg.get("conversation_model") or llm_cfg.get("tool_model", "qwen3:4b")
            base_url = llm_cfg.get("host", "http://localhost:11434")
            if base_url and not base_url.startswith("http"):
                base_url = f"http://{base_url}"
            llm = ChatOllama(model=model, base_url=base_url, temperature=0.7)
            resp = llm.invoke(
                [
                    SystemMessage(content="You are JARVIS. Brief, dry wit. Address user as Sir."),
                    HumanMessage(content=prompt),
                ]
            )
            return resp.content if hasattr(resp, "content") else str(resp)
        except Exception:
            return ""

    hb_cfg = config.get("heartbeat", {})
    interval = hb_cfg.get("interval_minutes", 30)
    try:
        from heartbeat import start_heartbeat

        start_heartbeat(
            memory,
            _tts_speak,
            _llm_invoke,
            interval_minutes=interval,
            allow_run=lambda: not session.is_active(),
        )
        if verbose:
            print(f"Heartbeat started (every {interval} min).", flush=True)
    except Exception as e:
        if verbose:
            print(f"Heartbeat not started: {e}", flush=True)

    if console_mode:
        print("Ready. Type your message below.", flush=True)
    elif verbose:
        print("Ready. Listening for wake word...", flush=True)

    threshold = wake_cfg.get("wake_confidence")
    if threshold is None:
        threshold = wake_cfg.get("threshold", 0.75)
    mic_device = wake_cfg.get("device")
    noise_gate_rms = wake_cfg.get("noise_gate_rms", 0.005)

    recorder = Recorder(
        sample_rate=sample_rate,
        silence_threshold=vc.get("recorder_silence_threshold", 0.012),
        silence_duration=vc.get("recorder_silence_duration", 2.5),
        device=mic_device,
        speech_start_threshold=vc.get("speech_start_threshold"),
    )

    try:
        from voice.wake import WakeWordDetector

        wake = WakeWordDetector(
            model_names=wake_cfg.get("models", ["hey_jarvis_v0.1"]),
            threshold=threshold,
            device=mic_device,
            wake_confidence=wake_cfg.get("wake_confidence"),
            noise_gate_rms=noise_gate_rms,
        )
        try:
            wake._ensure_model()
        except Exception:
            pass
    except ImportError:
        wake = None

    record_seconds = int(vc.get("push_to_talk_seconds", 5))
    wake_cooldown_sec = float(wake_cfg.get("cooldown_seconds", 3))
    _last_wake_time = [0.0]
    _processing = [False]

    def _signal_ready() -> None:
        if vc.get("ready_beep", True):
            _play_ready_beep(
                volume=float(vc.get("ready_beep_volume", 0.08)),
                seconds=float(vc.get("ready_beep_seconds", 0.08)),
                frequency_hz=float(vc.get("ready_beep_frequency_hz", 920)),
            )
        if not vc.get("ready_beep_only", False):
            listening_prompt = vc.get("listening_prompt", "Listening, Sir.")
            if listening_prompt:
                _speak(listening_prompt)

    def _transcribe_audio(audio: bytes, preferred_language: str | None = None) -> dict | None:
        if not audio:
            return None

        duration = len(audio) / float(sample_rate * 2)
        if duration < min_audio_len:
            if verbose:
                print(f"[Voice] Ignored short audio ({duration:.2f}s < {min_audio_len}s).", flush=True)
            return None

        arr = np.frombuffer(audio, dtype=np.int16)
        rms = np.sqrt(np.mean(arr.astype(np.float64) ** 2)) / 32768
        if rms <= min_rms:
            if verbose:
                print(f"[Voice] Ignored low RMS ({rms:.4f} <= {min_rms:.4f}).", flush=True)
            return None

        if use_vad:
            vad_ok, vad_reason = has_voice_activity(
                audio,
                sample_rate=sample_rate,
                min_speech_ratio=float(vc.get("vad_min_speech_ratio", 0.08)),
                vad_aggressiveness=int(vc.get("vad_aggressiveness", 2)),
            )
            if not vad_ok:
                if verbose:
                    print(f"[Voice] Rejected by pre-STT VAD ({vad_reason}).", flush=True)
                return None

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
            with wave.open(tmp_path, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                wav.writeframes(audio)
        try:
            lang = preferred_language or vc.get("stt_language") or "en"
            attempts = [lang]

            last_reason = "empty"
            t_stt = time.perf_counter()
            for lang in attempts:
                details = stt.transcribe_detailed(tmp_path, language=lang, vad_filter=use_vad)
                text = (details.get("text") or "").strip()
                end_commands = list(end_commands_cfg) if end_commands_cfg else []
                valid, reason = is_valid_transcript(
                    text, min_words=min_words, log_rejections=verbose,
                    allowed_short=end_commands,
                )
                if text and valid:
                    details["text"] = text
                    details["audio_duration"] = duration
                    details["rms"] = rms
                    if (config.get("timing", False) or config.get("debug", False)) and verbose:
                        stt_ms = (time.perf_counter() - t_stt) * 1000
                        from agent.timing_context import log_timing
                        log_timing("STT", stt_ms, config, verbose=True)
                    return details
                last_reason = reason or "invalid"
            # Don't show "Transcript rejected" to user - it's internal/debug only
            return None
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass

    def _run_turn(user_text: str, announce_thinking: bool = False) -> tuple[str, float]:
        try_open_browser_from_intent(user_text, tool_router)

        thinking_prompt = vc.get("thinking_prompt", "As you wish, Sir...")
        if announce_thinking and thinking_prompt:
            _speak(thinking_prompt)

        stream_tts = bool(vc.get("stream_tts", True))
        show_timing = config.get("timing", False) or config.get("debug", False)
        t_turn_start = time.perf_counter()

        if stream_tts:
            on_chunk, flush, put_remainder, start_consumer, stop_consumer = _make_streaming_tts_callback(
                tts,
                _play_audio_file,
                verbose,
            )
            consumer_thread = start_consumer()
            try:
                response, latency = invoke_jarvis(
                    graph,
                    user_text,
                    stream_callback=on_chunk,
                    max_words=max_words,
                    config=config,
                    memory=memory,
                )
                put_remainder(flush())
            finally:
                stop_consumer()
                consumer_thread.join(timeout=30)
        else:
            response, latency = invoke_jarvis(
                graph,
                user_text,
                stream_callback=None,
                max_words=max_words,
                config=config,
                memory=memory,
            )
            t_tts_start = time.perf_counter()
            _speak(response)
            if show_timing:
                tts_ms = (time.perf_counter() - t_tts_start) * 1000
                from agent.timing_context import log_timing
                log_timing("TTS+Playback", tts_ms, config, verbose=verbose)

        memory.save_interaction(user_text, response)
        if verbose:
            lat_str = f" [{latency:.1f}s]" if config.get("show_latency", True) else ""
            print(_display_text(response) + lat_str, flush=True)

        # Context tracking: show after each turn
        ctx_cfg = config.get("context", {})
        if ctx_cfg.get("show_after_each_turn"):
            from agent.timing_context import get_context_status
            disp, is_warning = get_context_status(config)
            if verbose:
                print(disp, flush=True)
            # Speak context if enabled (brief)
            if ctx_cfg.get("speak_context", False):
                _speak(disp)
            if is_warning:
                warning = "Sir, we're approaching my recollection limit. Shall I summarize and refresh?"
                if verbose:
                    print(warning, flush=True)
                _speak(warning)

        if show_timing:
            total_ms = (time.perf_counter() - t_turn_start) * 1000
            from agent.timing_context import log_timing
            log_timing("TurnTotal", total_ms, config, verbose=verbose)

        return response, latency

    def _run_active_session() -> None:
        session.activate()
        from agent.timing_context import reset_session_tokens
        reset_session_tokens()
        turn_index = 0
        wake_ack = vc.get("wake_ack_prompt", "")
        if wake_ack:
            _speak(wake_ack)

        try:
            if verbose:
                print("[Session] Active session started.", flush=True)
            while not stop_event.is_set() and session.is_active():
                remaining = session.time_remaining()
                if remaining <= 0:
                    if verbose:
                        print("[Session] Silence timeout reached; returning to wake mode.", flush=True)
                    break

                if verbose:
                    print(f"[Session] Listening... ({remaining:.1f}s remaining)", flush=True)
                start_timeout = min(max(0.6, speech_start_timeout), remaining)
                audio = recorder.record_utterance(start_timeout=start_timeout, max_seconds=max_record_seconds)
                if not audio:
                    if session.timed_out():
                        break
                    continue

                details = _transcribe_audio(audio)
                if not details:
                    continue

                text = (details.get("text") or "").strip()
                if not text:
                    continue

                session.touch()
                if verbose:
                    detected_lang = (details.get("language") or "en").lower()
                    print(f"[Voice] You said ({detected_lang}): {_display_text(text)}", flush=True)

                if session.should_end_for_text(text):
                    end_prompt = vc.get("session_end_prompt", "Very well, Sir. Standing by.")
                    if end_prompt:
                        _speak(end_prompt)
                    break

                _run_turn(text, announce_thinking=(turn_index == 0 or thinking_prompt_each_turn))
                turn_index += 1
                session.touch()
                _signal_ready()
        finally:
            session.end()
            if verbose:
                print("[Session] Wake-only mode.", flush=True)

    def on_wake() -> None:
        if session.is_active() or _processing[0]:
            return
        now = time.monotonic()
        if now - _last_wake_time[0] < wake_cooldown_sec:
            return
        _last_wake_time[0] = now
        _processing[0] = True

        def _process() -> None:
            try:
                if verbose:
                    print("[Wake] Detected. Entering active session...", flush=True)
                _run_active_session()
            except Exception as e:
                print(f"[Wake] Error: {e}", flush=True)
            finally:
                _processing[0] = False

        threading.Thread(target=_process, daemon=True).start()

    if push_to_talk and not console_mode:
        while not stop_event.is_set():
            try:
                input(f"Press Enter to speak ({record_seconds}s recording)... ")
                print("Listening...", flush=True)
                audio = recorder.record_fixed(seconds=record_seconds)
                details = _transcribe_audio(audio)
                if not details:
                    print("No valid speech detected. Try again.", flush=True)
                    continue
                text = details.get("text", "")
                print(f"You said: {_display_text(text)}", flush=True)
                print("JARVIS: ", end="", flush=True)
                _run_turn(text, announce_thinking=True)
                print()
            except (EOFError, KeyboardInterrupt):
                break
            except Exception as e:
                print(f"Error: {e}", flush=True)
    elif wake and not console_mode:
        show_scores = wake_cfg.get("show_scores", False) or "--debug-wake" in sys.argv
        if verbose:
            print("Listening for wake word: \"Hey Jarvis\"...", flush=True)
        if show_scores:
            print("Debug: showing wake scores every ~4s.", flush=True)
        try:
            wake.listen(on_detected=on_wake, verbose=verbose, show_scores=show_scores)
        except Exception as e:
            print(f"[Wake] Listener error: {e}", flush=True)
    else:
        while not stop_event.is_set():
            try:
                text = input("You: ").strip()
                if not text:
                    continue
                print("JARVIS: ", end="", flush=True)
                _run_turn(text, announce_thinking=False)
            except (EOFError, KeyboardInterrupt):
                break


def create_tray_icon(config: dict):
    """System tray with pystray."""
    import pystray
    from PIL import Image

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
        loop_thread = threading.Thread(
            target=run_jarvis_loop,
            args=(config, stop_event),
            kwargs={"console_mode": False},
            daemon=True,
        )
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

    # Console mode: text input, no tray.
    if "--console" in sys.argv or "-c" in sys.argv:
        print("JARVIS console mode. Type your message and press Enter. Ctrl+C to quit.")
        stop_event = threading.Event()
        run_jarvis_loop(config, stop_event, console_mode=True)
        return

    # Voice mode: wake word + voice directly, no tray.
    if "--mode" in sys.argv:
        idx = sys.argv.index("--mode")
        mode = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else ""
        if mode.lower() == "voice":
            push_to_talk = "--push-to-talk" in sys.argv or "-p" in sys.argv
            if push_to_talk:
                print("JARVIS push-to-talk. Press Enter to record. Ctrl+C to quit.")
            else:
                print("JARVIS voice mode. Say 'Hey Jarvis' then speak. Ctrl+C to quit.")
            stop_event = threading.Event()
            run_jarvis_loop(config, stop_event, console_mode=False, push_to_talk=push_to_talk)
            return

    # System tray mode (default).
    print("JARVIS tray mode. Look for the icon in your system tray (bottom-right).")
    print("Right-click the icon -> Start to begin. The terminal will stay open.", flush=True)
    icon = create_tray_icon(config)
    icon.run()


if __name__ == "__main__":
    main()
