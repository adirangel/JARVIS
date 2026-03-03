"""JARVIS — Main entry point.

Supports: --mode voice | cli | test
"""

import argparse
import asyncio
import os
import re
import sys
import yaml

# Suppress huggingface_hub symlinks warning on Windows
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from datetime import datetime
from loguru import logger

# ── Logging ───────────────────────────────────────────────────────────────────
# File: full debug trace.  Console: clean INFO-level only (loguru handles color).
logger.remove()
logger.add(
    "log/jarvis-{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="7 days",
    level="DEBUG",
    format="{time:HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
)
logger.add(
    sys.stderr,
    level="INFO",
    colorize=True,
    format="<level>[{level.name: <7}]</level> {message}",
)

from agent.graph import SessionState, AgentGraph
from agent.nodes import InputNode, ReflectorNode, ToolNode, OutputNode
from agent.personality import Personality
from agent.utils import color_print
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from voice.wake import WakeListener
from text_to_speech import TTS


# ── LLM helper for memory extraction / consolidation ─────────────────────────

def _make_llm_fn(config: dict):
    """Return a sync callable ``llm_fn(prompt) -> str`` for memory subsystem."""
    import httpx

    base_url = config.get("llm_base_url", "http://localhost:11434").rstrip("/v1").rstrip("/")
    model = config.get("llm_model", "llama3.2:3b")

    def llm_fn(prompt: str) -> str:
        """Call Ollama /api/generate for a short, non-streaming completion."""
        resp = httpx.post(
            f"{base_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False,
                   "options": {"temperature": 0.3, "num_predict": 512}},
            timeout=60.0,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

    return llm_fn


# ── Utility ───────────────────────────────────────────────────────────────────

def _strip_think_for_tts(text: str) -> str:
    """Remove <think>…</think> blocks from LLM output before speaking."""
    # Remove complete <think>...</think> tags
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # If an unclosed <think> tag remains, discard everything from it onward
    if "<think>" in cleaned:
        cleaned = cleaned[:cleaned.index("<think>")].strip()
    return cleaned


# ── Config ────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open("config/settings.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    # Load LLM API key
    key_file = cfg.get("llm_api_key_file", "api_key.txt")
    if os.path.exists(key_file):
        with open(key_file, "r") as f:
            cfg["llm_api_key"] = f.read().strip()
    else:
        cfg["llm_api_key"] = None
    # Load STT API key (falls back to llm_api_key)
    stt_key_file = cfg.get("stt_api_key_file", key_file)
    if os.path.exists(stt_key_file):
        with open(stt_key_file, "r") as f:
            cfg["stt_api_key"] = f.read().strip()
    else:
        cfg["stt_api_key"] = cfg.get("llm_api_key")
    return cfg


# ── Agent init ────────────────────────────────────────────────────────────────

async def initialize_agent(config: dict):
    session = SessionState(
        session_id=f"session_{int(datetime.now().timestamp())}",
        conversation_history=[],
        timers={},
        metadata={},
        speech_lock=asyncio.Lock(),
        stopped=asyncio.Event(),
    )
    session.config = config
    # Memories
    session.short_term_memory = ShortTermMemory(max_turns=config.get("max_conversation_turns", 15))
    if config.get("memory_long_term_enabled", False):
        try:
            llm_fn = _make_llm_fn(config)
            mem_cfg = {
                "db_path": config.get("memory_db_path", "data/jarvis.db"),
                "chroma_path": config.get("memory_chroma_path", "data/chroma"),
                "embedding_model": config.get("memory_embeddings_model", "nomic-embed-text"),
                "ollama_host": config.get("memory_ollama_host", "http://localhost:11434"),
            }
            session.long_term_memory = LongTermMemory({"memory": mem_cfg}, llm_fn=llm_fn)
            logger.info("Long-term memory enabled (forever mode)")
        except Exception as e:
            logger.warning(f"Long-term memory disabled: {e}")
            session.long_term_memory = None
    else:
        session.long_term_memory = None
    # Personality
    session.personality = Personality(config.get("personality", {}))
    # Build agent graph
    input_node = InputNode(name="input")
    reflector = ReflectorNode(name="reflector", config=config)
    tool_node = ToolNode(name="tool")
    output_node = OutputNode(name="output")
    graph = AgentGraph(entry_node=input_node)
    graph.add_node(reflector)
    graph.add_node(tool_node)
    graph.add_node(output_node)
    session.tts = None
    return session, graph


# ── Voice mode ────────────────────────────────────────────────────────────────

def _push_voice_status(status: str) -> None:
    """Fire-and-forget: notify the API of voice state change (listening/speaking/processing)."""
    try:
        import httpx
        httpx.post("http://localhost:8000/api/voice/status", json={"status": status}, timeout=1.0)
    except Exception:
        pass


async def handle_wake(
    transcript: str,
    session: SessionState,
    graph: AgentGraph,
    wake: WakeListener,
    config: dict,
):
    """Process a single voice turn: user said something → LLM → TTS → unmute mic."""
    logger.info(f"You: {transcript}")

    # Interrupt if already speaking
    if session.speech_lock.locked():
        logger.warning("Interrupting current response...")
        session.stopped.set()

    async with session.speech_lock:
        session.stopped.clear()
        session.metadata["audio_file"] = None
        session.current_input = transcript
        session.last_response = None
        session.streamed_to_console = False
        session.timers.clear()
        session.conversation_history.append({"role": "user", "content": transcript})

        # Mute mic while JARVIS speaks (prevent self-hearing / hallucinations)
        wake.mute()
        _push_voice_status("speaking")
        try:
            async for result in graph.run(session, max_turns=1):
                pass
        except asyncio.TimeoutError:
            logger.error("LLM response timed out (30s)")
        except Exception as e:
            err_msg = str(e) or type(e).__name__
            logger.error(f"Agent error: {err_msg}")
            if config.get("debug", False):
                import traceback
                traceback.print_exc()
        finally:
            wake.unmute()
            _push_voice_status("listening")

    session.current_input = ""


async def voice_loop(session: SessionState, graph: AgentGraph, config: dict):
    """Voice mode main loop."""
    wake = WakeListener(config)
    dev = config.get("input_device") or config.get("wake_device")
    dev_info = f"device={dev}" if dev is not None else "default mic"
    timeout_min = int(config.get("session_idle_timeout_seconds", 600)) // 60
    logger.info(f"Mic: {dev_info}. Say 'Hey Jarvis' to begin. {timeout_min} min silence = re-wake.")

    loop = asyncio.get_running_loop()

    def on_wake(transcript: str):
        asyncio.run_coroutine_threadsafe(
            handle_wake(transcript, session, graph, wake, config), loop
        )

    wake.start(callback=on_wake)
    logger.info("JARVIS is listening for wake word...")
    _push_voice_status("listening")

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        wake.stop()


# ── CLI mode ──────────────────────────────────────────────────────────────────

async def cli_loop(session: SessionState, graph: AgentGraph, config: dict):
    """Interactive console mode."""
    logger.info("JARVIS is ready. Type your messages below. Type 'exit' to stop.")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in ("exit", "quit", "q"):
            logger.info("Goodbye, Sir.")
            break
        if not user_input:
            continue

        async with session.speech_lock:
            session.current_input = user_input
            session.last_response = None
            session.streamed_to_console = False
            session.timers.clear()
            session.conversation_history.append({"role": "user", "content": user_input})
            try:
                async for result in graph.run(session, max_turns=1):
                    pass
            except Exception as e:
                logger.error(f"Agent error: {e}")
        session.current_input = ""


# ── Test mode ─────────────────────────────────────────────────────────────────

async def single_text_test(session: SessionState, graph: AgentGraph, text: str):
    """One-shot test: process a single user text and print response."""
    logger.info(f"Test input: {text}")
    async with session.speech_lock:
        session.current_input = text
        session.conversation_history.append({"role": "user", "content": text})
        try:
            async for result in graph.run(session, max_turns=1):
                pass
        except Exception as e:
            logger.error(f"Agent error: {e}")
    if session.last_response:
        logger.success(f"JARVIS: {session.last_response}")
    else:
        logger.error("No response generated.")


# ── Entry point ───────────────────────────────────────────────────────────────

async def main(args):
    config = load_config()
    logger.info("Starting JARVIS")
    session, graph = await initialize_agent(config)

    # ── Schedule heartbeat with memory consolidation ──────────────────────────
    if session.long_term_memory is not None:
        from heartbeat import heartbeat_job
        import threading

        def _heartbeat():
            """30-minute heartbeat loop in a daemon thread."""
            import time as _t
            interval = config.get("proactive_interval_minutes", 30) * 60
            while True:
                _t.sleep(interval)
                try:
                    llm_invoke = _make_llm_fn(config) if not config.get("llm_mock") else None
                    heartbeat_job(
                        memory=session.long_term_memory,
                        tts_speak=lambda text: logger.info(f"[Heartbeat] {text}"),
                        llm_invoke=llm_invoke,
                    )
                except Exception as e:
                    logger.debug(f"Heartbeat error: {e}")

        t = threading.Thread(target=_heartbeat, daemon=True)
        t.start()
        logger.info("Memory heartbeat scheduled (consolidation + tasks)")

    if args.mode == "voice":
        await voice_loop(session, graph, config)
    elif args.mode == "cli":
        await cli_loop(session, graph, config)
    elif args.mode == "test":
        if args.test_text:
            await single_text_test(session, graph, args.test_text)
        else:
            logger.error("--test-text required when mode=test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JARVIS AI Assistant")
    parser.add_argument(
        "--mode",
        default="voice",
        choices=["voice", "cli", "test"],
        help="voice (wake word) | cli (console) | test (single message)",
    )
    parser.add_argument("--test-text", help="For mode=test, provide the text input.")
    args = parser.parse_args()
    asyncio.run(main(args))
