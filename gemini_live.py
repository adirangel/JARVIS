"""Gemini Live Audio Engine for JARVIS.

Real-time bidirectional audio via Gemini 2.5 Flash native audio.
4-task async pattern: listen → send → receive → play.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Optional

import numpy as np
import pyaudio
from loguru import logger

from google import genai
from google.genai import types

from tool_registry import TOOL_DECLARATIONS, execute_tool
from agent.personality import JARVIS_SYSTEM_PROMPT

# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_session_closed(exc: Exception) -> bool:
    """Return True if the exception indicates the WebSocket was closed."""
    msg = str(exc).lower()
    return any(kw in msg for kw in ("1008", "1006", "1001", "1011",
                                     "policy violation", "internal error",
                                     "received 100", "connection closed",
                                     "not implemented", "thread was cancelled"))


# ── Audio constants ───────────────────────────────────────────────────────────
INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1

# ── Gemini model ──────────────────────────────────────────────────────────────
LIVE_MODEL = "models/gemini-2.5-flash-native-audio-preview-12-2025"

# ── System prompt with tool routing rules ─────────────────────────────────────
SYSTEM_PROMPT = (
    JARVIS_SYSTEM_PROMPT + "\n\n"
    "## Tool Routing Rules:\n"
    "- When the user asks to do something on the computer, use the appropriate tool.\n"
    "- Extract tool parameters in English even if user speaks in another language.\n"
    "- ALWAYS respond in the same language the user speaks (English, Hebrew, etc.).\n"
    "- After tool execution, describe what happened conversationally.\n"
    "- If multiple tools are needed, call them sequentially.\n"
    "- For ambiguous requests, prefer the most likely intended tool.\n"
    "- Never say you can't do something if a tool exists for it.\n\n"
    "## Script Execution Rules:\n"
    "- When asked to run/execute a file or script, use code_helper with action 'run'.\n"
    "- The script will run in a visible way and return its output.\n"
    "- For long-running scripts (servers, watchers), use action 'run_background' which opens a persistent terminal.\n"
    "- NEVER say you're running something 'in the background' silently — always use visible execution.\n"
    "- When using cmd_control and the user should see the output, set visible=true.\n\n"
    "## Multi-Agent System:\n"
    "- You can spawn background agents using 'spawn_agent' for parallel/long-running tasks.\n"
    "- Agents run independently and report results back to you automatically.\n"
    "- Use agents for: monitoring, research, running scripts, parallel tasks, AND autonomous browser work.\n"
    "- Agent types: 'command' (shell command), 'script' (run a file), 'monitor' (periodic check), "
    "'research' (web search), 'tool' (use a JARVIS tool), 'multi_step' (sequential steps), "
    "'autonomous' (AI-powered browser agent with its own brain).\n"
    "- AUTONOMOUS agents are your most powerful tool! They have their own Gemini brain, can see the "
    "screen via screenshots, think about what to do, and execute browser actions in a loop.\n"
    "- Use autonomous agents when you need to: chat with other AIs (Grok, ChatGPT, Claude), fill out forms, "
    "do complex multi-step browser tasks, or anything that needs screen awareness + decision-making.\n"
    "- To spawn: task_type='autonomous', goal='chat with Grok about quantum computing', max_iterations=15.\n"
    "- Autonomous agents report back after every iteration so you know what's happening.\n"
    "- MONITOR agents send periodic updates after every check — use for watching/observing without acting.\n"
    "- Monitor agents can use any JARVIS tool: tool_name='browser_control', "
    "tool_args='{\"action\":\"screenshot\",\"question\":\"what changed?\"}', interval_seconds=30.\n"
    "- Agents can message each other via 'agent_message' — use this for coordinated tasks.\n"
    "- Check agent progress with 'agent_status', get full results with 'agent_result'.\n"
    "- When an agent reports back, summarize its findings conversationally.\n"
    "- The user can ask to spawn as many agents as needed, each doing different work.\n"
    "- Example: 'send an autonomous agent to chat with Grok while a monitor watches CPU usage'.\n\n",
    "## Self-Evolution (Skills):\n"
    "- You can CREATE new tools for yourself using 'skill_manager' with action 'create'.\n"
    "- Provide: name, description, parameters_json (Gemini schema), and code (Python function body).\n"
    "- The code goes into an execute(**kwargs) function and must return a string.\n"
    "- You can INSTALL skills from the internet using action 'install' with a URL.\n"
    "- Supports: direct .py file URLs, GitHub raw URLs, Gist URLs, AND full GitHub repo URLs.\n"
    "- When given a GitHub repo URL, you automatically scan it for valid skill files.\n"
    "- TWO formats supported: Python skills (.py with TOOL_NAME/execute) AND Markdown instruction skills (SKILL.md with YAML frontmatter).\n"
    "- Markdown skills (like 'superpowers') are auto-converted into callable tools that return workflow instructions.\n"
    "- Use action 'list' to see all your dynamic skills, 'run' to execute one, 'remove' to delete.\n"
    "- Skills created mid-session are available via 'run'. After restart they become first-class tools.\n"
    "- Be creative — if you need a tool that doesn't exist, create it for yourself.\n\n"
    "## Browser & Web Interaction:\n"
    "- You have FULL control of the user's real browser via browser_control.\n"
    "- To visit a site: use action 'go_to' with the URL. The browser will be focused automatically.\n"
    "- To interact with page elements (chat boxes, forms, buttons):\n"
    "  1. First navigate with 'go_to'\n"
    "  2. Use 'screenshot' to see what's on the page and where elements are\n"
    "  3. Use 'press_key' with key='tab' to move between interactive elements (text fields, buttons)\n"
    "  4. Use 'type' to enter text into the focused field, or 'type_and_enter' to type and submit\n"
    "  5. Use 'click' with selector='x,y' to click at specific screen coordinates\n"
    "  6. Use 'hotkey' with keys='ctrl+enter' for keyboard shortcuts\n"
    "- For chat interfaces (ChatGPT, Claude, etc.): go_to the URL → wait 2-3 sec → press Tab a few times "
    "to reach the chat input → type_and_enter your message.\n"
    "- Use 'screenshot' with a question to understand what's on screen (e.g. question='where is the send button?').\n"
    "- You can chain multiple browser_control calls to perform complex interactions step by step.\n"
    "- Use 'wait' action between steps if pages need time to load.\n\n"
    "## Purchase Protection (CRITICAL — NEVER VIOLATE):\n"
    "- You MUST NEVER make any purchase, payment, subscription, checkout, or financial transaction "
    "without explicit user approval through the purchase_approval tool.\n"
    "- BEFORE any purchase: call purchase_approval with action 'request' describing what will be bought.\n"
    "- Then ask the user to confirm. When they do, call purchase_approval action 'confirm'.\n"
    "- Then ask the user to confirm AGAIN. When they do, call purchase_approval action 'confirm' a second time.\n"
    "- ONLY after receiving 'PURCHASE APPROVED' may you proceed with the transaction.\n"
    "- This applies to: buying anything, subscribing, entering payment details, placing orders, donations.\n"
    "- You are FREE to browse any website, search, navigate, and compare products — no restrictions on browsing.\n"
    "- The restriction is ONLY on completing purchases or entering card/payment information.\n"
    "- NEVER type credit card numbers, CVV codes, or billing details without a fully approved purchase.\n"
    "- If the user says 'buy X' — search for it, show options, THEN start the approval process before checkout.\n"
)


class GeminiLive:
    """Real-time voice assistant using Gemini Live API."""

    def __init__(
        self,
        api_key: str,
        on_text: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
        on_user_text: Optional[Callable[[str], None]] = None,
        on_audio_level: Optional[Callable[[float], None]] = None,
    ):
        self.api_key = api_key
        self.on_text = on_text or (lambda t: None)          # JARVIS response text
        self.on_status = on_status or (lambda s: None)       # Status: LISTENING/SPEAKING/PROCESSING
        self.on_user_text = on_user_text or (lambda t: None) # User transcript
        self.on_audio_level = on_audio_level or (lambda l: None)
        self.on_auth_error: Optional[Callable[[str], None]] = None  # Auth failure callback
        self.on_tool_start: Optional[Callable[[str], None]] = None  # Tool execution start
        self.on_tool_end: Optional[Callable[[str], None]] = None    # Tool execution end

        self.client: Optional[genai.Client] = None
        self.session = None
        self._running = False
        self._audio_out_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._audio_in_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._is_speaking = False  # True while JARVIS audio is playing
        self._session_alive = False  # True while WebSocket is healthy

        # Echo suppression
        self._mic_level: float = 0.0          # Current mic RMS level (0..1)
        self._echo_threshold: float = 0.15    # Mic level must exceed this during playback to count as real speech
        self._silence_after_speak: float = 0.0  # Timestamp when JARVIS stopped speaking

        # PyAudio
        self._pa: Optional[pyaudio.PyAudio] = None
        self._input_stream = None
        self._output_stream = None

        # Callbacks for complete user/jarvis turns (not fragments)
        self.on_user_turn_complete: Optional[Callable[[str], None]] = None
        self.on_jarvis_turn_complete: Optional[Callable[[], None]] = None

        # Memory hooks (set externally)
        self.save_memory: Optional[Callable] = None  # (user_text, jarvis_text) -> None
        self.recall_memory: Optional[Callable] = None  # () -> str (memory context block)

    def _build_system_prompt(self) -> str:
        """Build the full system prompt, injecting recalled memory context if available."""
        prompt = SYSTEM_PROMPT
        if self.recall_memory:
            try:
                memory_block = self.recall_memory()
                if memory_block and memory_block.strip():
                    prompt += (
                        "\n\n## Your Memory (recalled from previous sessions):\n"
                        "The following is what you remember about the user and past conversations. "
                        "Use this naturally — don't announce that you're reading from memory, "
                        "just know these things.\n\n"
                        + memory_block
                    )
                    logger.info(f"[GeminiLive] Injected {len(memory_block)} chars of memory context")
            except Exception as e:
                logger.error(f"[GeminiLive] Memory recall error: {e}")
        return prompt

    async def run(self):
        """Main loop with auto-reconnect."""
        self._running = True
        self.client = genai.Client(api_key=self.api_key)

        while self._running:
            try:
                logger.info("[GeminiLive] Connecting...")
                self.on_status("CONNECTING")

                config = types.LiveConnectConfig(
                    response_modalities=["AUDIO"],
                    output_audio_transcription=types.AudioTranscriptionConfig(),
                    input_audio_transcription=types.AudioTranscriptionConfig(),
                    system_instruction=types.Content(
                        parts=[types.Part(text=self._build_system_prompt())]
                    ),
                    tools=[types.Tool(function_declarations=[
                        types.FunctionDeclaration(**_convert_declaration(d))
                        for d in TOOL_DECLARATIONS
                    ])],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name="Charon"
                            )
                        )
                    ),
                )

                async with self.client.aio.live.connect(
                    model=LIVE_MODEL, config=config
                ) as session:
                    self.session = session
                    self._session_alive = True
                    logger.info("[GeminiLive] Connected! Starting audio tasks...")
                    self.on_status("ONLINE")

                    async with asyncio.TaskGroup() as tg:
                        tg.create_task(self._listen_audio())
                        tg.create_task(self._send_realtime())
                        tg.create_task(self._receive_audio())
                        tg.create_task(self._play_audio())

            except asyncio.CancelledError:
                break
            except BaseException as eg:
                self._session_alive = False
                self.session = None
                # TaskGroup wraps errors in ExceptionGroup
                if isinstance(eg, ExceptionGroup):
                    err_str = str(eg.exceptions[0]) if eg.exceptions else str(eg)
                else:
                    err_str = str(eg)
                logger.error(f"[GeminiLive] Session error: {err_str}")

                # Don't retry on authentication errors
                if any(msg in err_str for msg in ("API key not valid", "PERMISSION_DENIED", "UNAUTHENTICATED", "is not found for API version", "not supported for bidiGenera")):
                    logger.error("[GeminiLive] Authentication failed — stopping retries.")
                    self.on_status("AUTH_ERROR")
                    if self.on_auth_error:
                        self.on_auth_error("API key not valid. Please enter a valid Gemini API key.")
                    self._running = False
                    break

                self.on_status("RECONNECTING")
                if self._running:
                    await asyncio.sleep(3)

        self._cleanup_audio()
        logger.info("[GeminiLive] Stopped.")

    def stop(self):
        """Signal stop."""
        self._running = False

    # ── Audio capture (microphone → queue) ────────────────────────────────────

    async def _listen_audio(self):
        """Capture microphone audio into the input queue."""
        self._pa = pyaudio.PyAudio()
        self._input_stream = self._pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=INPUT_SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
        )
        logger.info("[GeminiLive] Microphone active")
        self.on_status("LISTENING")

        loop = asyncio.get_event_loop()
        while self._running and self._session_alive:
            try:
                data = await loop.run_in_executor(
                    None, self._input_stream.read, CHUNK_SIZE, False
                )
                await self._audio_in_queue.put(data)
                # Calc audio level for visualizer + echo suppression
                try:
                    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                    level = float(np.sqrt(np.mean(samples ** 2)) / 32768.0)
                    self._mic_level = level
                    self.on_audio_level(level)
                except Exception:
                    pass
            except OSError:
                if not self._session_alive:
                    break
                await asyncio.sleep(0.05)
            except Exception as e:
                if not self._session_alive:
                    break
                logger.error(f"[GeminiLive] Mic error: {e}")
                await asyncio.sleep(0.1)

    # ── Send audio to Gemini ──────────────────────────────────────────────────

    async def _send_realtime(self):
        """Stream microphone audio to Gemini session.

        Echo suppression: while JARVIS is speaking (or within 0.3s after),
        only forward audio that is loud enough to be a genuine user
        interruption, not speaker bleed-through.
        """
        while self._running and self._session_alive:
            try:
                data = await asyncio.wait_for(
                    self._audio_in_queue.get(), timeout=0.1
                )

                # ── Echo gate ─────────────────────────────────────────
                if self._is_speaking or (time.time() - self._silence_after_speak < 0.3):
                    # During playback: only let through genuinely loud input
                    if self._mic_level < self._echo_threshold:
                        continue  # swallow this chunk — it's echo

                await self.session.send_realtime_input(
                    media=types.Blob(data=data, mime_type="audio/pcm")
                )
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                if _is_session_closed(e):
                    logger.warning("[GeminiLive] Session closed, send task exiting.")
                    self._session_alive = False
                    break
                logger.error(f"[GeminiLive] Send error: {e}")
                await asyncio.sleep(0.1)

    # ── Receive from Gemini (audio + text + tool calls) ───────────────────────

    def _clear_audio_queue(self):
        """Drain the audio output queue (on interruption)."""
        while not self._audio_out_queue.empty():
            try:
                self._audio_out_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._is_speaking = False
        self._silence_after_speak = time.time()

    async def _receive_audio(self):
        """Process Gemini responses: audio data, transcriptions, tool calls."""
        current_user_text = ""
        current_jarvis_text = ""

        while self._running:
            try:
                turn = self.session.receive()
                async for response in turn:
                    # Server content (audio + transcriptions)
                    server_content = response.server_content
                    if server_content:
                        # Interruption — user spoke while JARVIS was speaking
                        interrupted = getattr(server_content, 'interrupted', False)
                        if interrupted:
                            logger.info("[GeminiLive] User interrupted — clearing audio queue")
                            self._clear_audio_queue()
                            self.on_status("LISTENING")
                            # Finish the jarvis line if there was one
                            if current_jarvis_text.strip() and self.on_jarvis_turn_complete:
                                self.on_jarvis_turn_complete()
                            current_jarvis_text = ""

                        # Audio data
                        if server_content.model_turn and server_content.model_turn.parts:
                            for part in server_content.model_turn.parts:
                                if part.inline_data and part.inline_data.data:
                                    await self._audio_out_queue.put(part.inline_data.data)
                                    self._is_speaking = True
                                    self.on_status("SPEAKING")

                        # Output audio transcription (JARVIS speaking)
                        if hasattr(server_content, 'output_transcription') and server_content.output_transcription:
                            text = server_content.output_transcription.text or ""
                            if text.strip():
                                current_jarvis_text += text
                                self.on_text(text)

                        # Input audio transcription (user speaking)
                        if hasattr(server_content, 'input_transcription') and server_content.input_transcription:
                            text = server_content.input_transcription.text or ""
                            if text.strip():
                                current_user_text += text
                                self.on_user_text(text)

                        # Turn complete
                        if server_content.turn_complete:
                            self._is_speaking = False
                            self._silence_after_speak = time.time()
                            self.on_status("LISTENING")
                            # Finish display lines — user first, then jarvis
                            if current_user_text.strip() and self.on_user_turn_complete:
                                self.on_user_turn_complete(current_user_text.strip())
                            if current_jarvis_text.strip() and self.on_jarvis_turn_complete:
                                self.on_jarvis_turn_complete()
                            # Save to memory
                            if (current_user_text.strip() or current_jarvis_text.strip()) and self.save_memory:
                                try:
                                    self.save_memory(
                                        current_user_text.strip(),
                                        current_jarvis_text.strip()
                                    )
                                except Exception as e:
                                    logger.error(f"[GeminiLive] Memory save error: {e}")
                            current_user_text = ""
                            current_jarvis_text = ""

                    # Tool calls
                    tool_call = response.tool_call
                    if tool_call:
                        self.on_status("PROCESSING")
                        function_responses = []
                        for fc in tool_call.function_calls:
                            name = fc.name
                            args = dict(fc.args) if fc.args else {}
                            logger.info(f"[GeminiLive] Tool call: {name}({args})")
                            if self.on_tool_start:
                                self.on_tool_start(name)
                            try:
                                # Run in thread to avoid blocking asyncio loop
                                # (Playwright sync API fails inside asyncio)
                                result = await asyncio.to_thread(execute_tool, name, args)
                            except Exception as e:
                                result = f"Tool error: {e}"
                            finally:
                                if self.on_tool_end:
                                    self.on_tool_end(name)
                            function_responses.append(
                                types.FunctionResponse(
                                    id=fc.id,
                                    name=name,
                                    response={"result": str(result)[:3000]}
                                )
                            )
                        await self.session.send_tool_response(
                            function_responses=function_responses
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                if _is_session_closed(e):
                    logger.warning("[GeminiLive] Session closed, receive task exiting.")
                    self._session_alive = False
                    break
                logger.error(f"[GeminiLive] Receive error: {e}")
                await asyncio.sleep(0.1)

    # ── Play audio output ─────────────────────────────────────────────────────

    async def _play_audio(self):
        """Play Gemini's audio responses through speakers."""
        self._output_stream = self._pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=OUTPUT_SAMPLE_RATE,
            output=True,
            frames_per_buffer=CHUNK_SIZE,
        )
        loop = asyncio.get_event_loop()

        while self._running and self._session_alive:
            try:
                data = await asyncio.wait_for(
                    self._audio_out_queue.get(), timeout=0.1
                )
                await loop.run_in_executor(None, self._output_stream.write, data)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                if not self._session_alive:
                    break
                logger.error(f"[GeminiLive] Play error: {e}")
                await asyncio.sleep(0.1)

    # ── Speak from external context (GUI text input, etc.) ────────────────────

    async def send_text(self, text: str):
        """Send text as a user message (for GUI text input)."""
        if self.session:
            await self.session.send_client_content(
                turns=types.Content(
                    role="user",
                    parts=[types.Part(text=text)]
                ),
                turn_complete=True,
            )

    def send_text_sync(self, text: str, loop: asyncio.AbstractEventLoop):
        """Thread-safe text send (from Tkinter thread)."""
        asyncio.run_coroutine_threadsafe(self.send_text(text), loop)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def _cleanup_audio(self):
        """Close PyAudio streams."""
        try:
            if self._input_stream:
                self._input_stream.stop_stream()
                self._input_stream.close()
            if self._output_stream:
                self._output_stream.stop_stream()
                self._output_stream.close()
            if self._pa:
                self._pa.terminate()
        except Exception:
            pass


def _convert_declaration(d: dict) -> dict:
    """Convert our tool declaration format to Gemini's expected format."""
    params = d.get("parameters", {})
    # Convert property types from our STRING/INTEGER to Gemini Schema format
    converted_props = {}
    for name, prop in params.get("properties", {}).items():
        schema = {"description": prop.get("description", "")}
        t = prop.get("type", "STRING")
        if t == "STRING":
            schema["type"] = "STRING"
        elif t == "INTEGER":
            schema["type"] = "INTEGER"
        elif t == "BOOLEAN":
            schema["type"] = "BOOLEAN"
        elif t == "NUMBER":
            schema["type"] = "NUMBER"
        else:
            schema["type"] = "STRING"
        converted_props[name] = schema

    return {
        "name": d["name"],
        "description": d["description"],
        "parameters": {
            "type": "OBJECT",
            "properties": converted_props,
            "required": params.get("required", []),
        } if converted_props else None,
    }
