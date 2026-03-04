"""Gemini Live Audio Engine for JARVIS.

Real-time bidirectional audio via Gemini 2.5 Flash native audio.
4-task async pattern: listen → send → receive → play.
"""

from __future__ import annotations

import asyncio
import base64
import json
import struct
import traceback
from typing import Any, Callable, Optional

import numpy as np
import pyaudio
from loguru import logger

from google import genai
from google.genai import types

from tool_registry import TOOL_DECLARATIONS, execute_tool
from agent.personality import JARVIS_SYSTEM_PROMPT

# ── Audio constants ───────────────────────────────────────────────────────────
INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1

# ── Gemini model ──────────────────────────────────────────────────────────────
LIVE_MODEL = "models/gemini-2.5-flash-native-audio-latest"

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
    "- Never say you can't do something if a tool exists for it.\n"
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

        # PyAudio
        self._pa: Optional[pyaudio.PyAudio] = None
        self._input_stream = None
        self._output_stream = None

        # Callbacks for complete user/jarvis turns (not fragments)
        self.on_user_turn_complete: Optional[Callable[[str], None]] = None
        self.on_jarvis_turn_complete: Optional[Callable[[], None]] = None

        # Memory hooks (set externally)
        self.save_memory: Optional[Callable] = None  # (user_text, jarvis_text) -> None

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
                        parts=[types.Part(text=SYSTEM_PROMPT)]
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
                    logger.info("[GeminiLive] Connected! Starting audio tasks...")
                    self.on_status("ONLINE")

                    async with asyncio.TaskGroup() as tg:
                        tg.create_task(self._listen_audio())
                        tg.create_task(self._send_realtime())
                        tg.create_task(self._receive_audio())
                        tg.create_task(self._play_audio())

            except asyncio.CancelledError:
                break
            except Exception as e:
                err_str = str(e)
                logger.error(f"[GeminiLive] Session error: {e}\n{traceback.format_exc()}")

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
        while self._running:
            try:
                data = await loop.run_in_executor(
                    None, self._input_stream.read, CHUNK_SIZE, False
                )
                await self._audio_in_queue.put(data)
                # Calc audio level for visualizer
                try:
                    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                    level = float(np.sqrt(np.mean(samples ** 2)) / 32768.0)
                    self.on_audio_level(level)
                except Exception:
                    pass
            except OSError:
                await asyncio.sleep(0.05)
            except Exception as e:
                logger.error(f"[GeminiLive] Mic error: {e}")
                await asyncio.sleep(0.1)

    # ── Send audio to Gemini ──────────────────────────────────────────────────

    async def _send_realtime(self):
        """Stream microphone audio to Gemini session."""
        while self._running:
            try:
                data = await asyncio.wait_for(
                    self._audio_in_queue.get(), timeout=0.1
                )
                await self.session.send_realtime_input(
                    media=types.Blob(data=data, mime_type="audio/pcm")
                )
            except asyncio.TimeoutError:
                continue
            except Exception as e:
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

        while self._running:
            try:
                data = await asyncio.wait_for(
                    self._audio_out_queue.get(), timeout=0.1
                )
                await loop.run_in_executor(None, self._output_stream.write, data)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
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
