"""JARVIS — Main entry point.

Simple: GUI → API key → Gemini Live on daemon thread → Tkinter mainloop.
"""

import asyncio
import os
import sys
import threading

# ── Logging ───────────────────────────────────────────────────────────────────
from loguru import logger

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
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
)

from gui import JarvisGUI
from gemini_live import GeminiLive

# ── Async event loop reference (shared between threads) ──────────────────────
_loop: asyncio.AbstractEventLoop = None


def start_gemini(api_key: str, gui: JarvisGUI):
    """Launch GeminiLive in a daemon thread."""
    global _loop

    engine = GeminiLive(
        api_key=api_key,
        on_text=lambda t: gui.append_token(t),
        on_status=lambda s: gui.set_status(s),
        on_user_text=lambda t: gui.append_user_token(t),
        on_audio_level=lambda l: gui.set_audio_level(l),
    )

    # Complete-line callbacks (newline after streaming)
    engine.on_user_turn_complete = lambda text: gui.finish_user_line()
    engine.on_jarvis_turn_complete = lambda: gui.finish_jarvis_line()

    # On auth failure, show API key screen again
    engine.on_auth_error = lambda msg: gui.show_api_key_screen(msg)

    # Tool execution tracking for Mission Control
    engine.on_tool_start = lambda name: gui.set_active_tool(name)
    engine.on_tool_end = lambda name: gui.clear_active_tool(name)

    # Wire memory (optional — if memory module is available)
    try:
        from memory.long_term import LongTermMemory

        # Create Gemini-based embedding function
        gemini_client = None
        def gemini_embed(text: str) -> list[float]:
            nonlocal gemini_client
            if gemini_client is None:
                from google import genai
                gemini_client = genai.Client(api_key=api_key)
            result = gemini_client.models.embed_content(
                model="models/text-embedding-004",
                contents=text,
            )
            return result.embeddings[0].values

        # Create Gemini-based LLM function for fact extraction
        def gemini_llm(prompt: str) -> str:
            nonlocal gemini_client
            if gemini_client is None:
                from google import genai
                gemini_client = genai.Client(api_key=api_key)
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash-preview-05-20",
                contents=prompt,
            )
            return response.text or ""

        long_term = LongTermMemory(
            config={
                "db_path": "data/jarvis.db",
                "chroma_path": "data/chroma",
            },
            llm_fn=gemini_llm,
            embed_fn=gemini_embed,
        )

        def save_mem(user_text: str, jarvis_text: str):
            if user_text or jarvis_text:
                try:
                    long_term.save_interaction(user_text, jarvis_text)
                    logger.debug(f"[Memory] Saved: user={user_text[:50]}...")
                except Exception as e:
                    logger.error(f"[Memory] Save error: {e}")

        engine.save_memory = save_mem
        gui.set_subagent_status("Memory", "active", "Long-term memory connected")
        gui.log_activity("Long-term memory module loaded", "system")
        logger.info("[Memory] Long-term memory connected.")
    except Exception as e:
        gui.set_subagent_status("Memory", "idle", "Not available")
        logger.warning(f"[Memory] Not available: {e}")

    # Text input from GUI → send to Gemini
    def on_text(text: str):
        if _loop and engine.session:
            engine.send_text_sync(text, _loop)

    gui.on_text_input(on_text)

    # ── Agent Manager setup ───────────────────────────────────────────────────
    try:
        from agent.agent_manager import AgentManager

        agent_mgr = AgentManager()

        # When an agent's status changes → update the GUI panels
        def _on_agent_status(name: str, status: str, desc: str):
            gui.set_subagent_status(f"Agent:{name}", status, desc)
            gui.log_agent_activity(name, desc, "ok" if status == "completed" else ("error" if status == "failed" else "info"))
            if status == "running":
                gui.log_activity(f"Agent '{name}' started", "agent")
            elif status == "completed":
                gui.log_activity(f"Agent '{name}' completed", "agent")
            elif status == "failed":
                gui.log_activity(f"Agent '{name}' failed: {desc[:50]}", "agent")

        agent_mgr.on_agent_status = _on_agent_status

        # When an agent reports a result → send it to JARVIS via Gemini session
        def _on_agent_result(agent_name: str, result: str):
            if _loop and engine.session:
                msg = (
                    f"[Agent Report] Agent '{agent_name}' has completed its task. "
                    f"Result: {result[:2000]}"
                )
                asyncio.run_coroutine_threadsafe(
                    engine.send_text(msg), _loop
                )
                gui.log_activity(f"Agent '{agent_name}' reported result", "agent")

        agent_mgr.on_agent_result = _on_agent_result

        # Wire live progress updates from agents
        def _on_agent_progress(agent_name: str, message: str):
            gui.log_agent_activity(agent_name, message, "info")

        agent_mgr.on_agent_progress = _on_agent_progress

        gui.set_subagent_status("Agents", "active", "Agent manager ready")
        gui.log_activity("Multi-agent system loaded", "system")
        logger.info("[Agents] Agent manager connected.")
    except Exception as e:
        gui.set_subagent_status("Agents", "idle", "Not available")
        logger.warning(f"[Agents] Not available: {e}")

    def _run():
        global _loop
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
        _loop.run_until_complete(engine.run())

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    logger.info("[Main] Gemini Live engine started on daemon thread.")


def main():
    """Entry point."""
    os.makedirs("log", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("config", exist_ok=True)

    logger.info("=" * 50)
    logger.info("  J.A.R.V.I.S  —  Starting up...")
    logger.info("=" * 50)

    gui = JarvisGUI()

    # If API key already loaded, start immediately
    if gui.api_key:
        logger.info("[Main] API key found, starting Gemini Live...")
        start_gemini(gui.api_key, gui)
    else:
        logger.info("[Main] Waiting for API key...")

    # When API key is entered on the GUI screen
    gui.on_api_key_ready(lambda key: start_gemini(key, gui))

    # Run Tkinter main loop (blocks until window closed)
    gui.run()

    logger.info("[Main] JARVIS shutdown.")


if __name__ == "__main__":
    main()
