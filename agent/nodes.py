import asyncio
import os
import queue
import threading
import time
from typing import Optional, Dict, Any, List

from loguru import logger

from agent.graph import Node, SessionState
from agent.utils import DebugTimer, color_print
from agent.personality import Personality
from audio_to_text import transcribe
from text_to_speech import TTS
from tools.computer_control import *
from tools.system_monitor import get_system_summary, format_for_speech
from memory.long_term import LongTermMemory

try:
    import yaml
except ImportError:
    yaml = None

class InputNode(Node):
    async def process(self, state: SessionState) -> Optional[str]:
        color_print('thought', 'InputNode: Listening...')
        # Preserve input already set (e.g. from wake word)
        if getattr(state, 'current_input', None):
            return state.current_input
        audio_file = state.metadata.get('audio_file')
        if not audio_file:
            state.current_input = "Hey Jarvis, what's the time?"  # test fallback
            return state.current_input
        api_key = state.config.get('stt_api_key') or state.config.get('llm_api_key')
        text = transcribe(audio_file, api_key)
        color_print('info', f"User said: {text}")
        state.conversation_history.append({"role": "user", "content": text})
        state.current_input = text
        return text

class ReflectorNode(Node):
    # Class-level cache for OpenAI client (reused across all requests)
    _client_cache: Dict[str, Any] = {}
    _tts_cache: Dict[str, TTS] = {}
    
    def __init__(self, name: str = "reflector", config: Optional[Dict[str, Any]] = None):
        super().__init__(name)
        self.config = config or {}
        self.personality = Personality(self.config.get('personality', {}))
        self.tts: Optional[TTS] = None
        self.max_tokens = self.config.get('llm_max_tokens', 500)
        self.streaming = self.config.get('streaming', True)
        self.api_key = self.config.get('llm_api_key')
        self.provider = self.config.get('llm_provider', 'stepfun')
        self._initialized = False
    
    async def initialize(self):
        """Initialize TTS only once, and skip if engine is 'none' or in CLI mode."""
        if self._initialized:
            return
        self._initialized = True
        engine = self.config.get('tts_engine', '').lower()
        # Skip TTS if disabled or in CLI-only mode
        if not engine or engine == 'none':
            return
        # Use cached TTS if available
        cache_key = f"{engine}_{self.config.get('tts_quality', 'medium')}"
        if cache_key in ReflectorNode._tts_cache:
            self.tts = ReflectorNode._tts_cache[cache_key]
            return
        if self.streaming:
            self.tts = TTS(self.config)
            await self.tts.initialize()
            ReflectorNode._tts_cache[cache_key] = self.tts
    
    def _get_client(self):
        """Get or create cached OpenAI client for LLM calls."""
        base_url = self.config.get('llm_base_url')
        if not base_url:
            base_url = "https://api.stepfun.com/v1" if self.provider == "stepfun" else "https://openrouter.ai/api/v1"
        api_key = self.api_key
        if self.provider == "ollama" or (base_url and "localhost:11434" in base_url):
            api_key = api_key or "ollama"
        
        cache_key = f"{base_url}_{api_key}"
        if cache_key not in ReflectorNode._client_cache:
            from openai import OpenAI
            ReflectorNode._client_cache[cache_key] = OpenAI(
                api_key=api_key or "ollama",
                base_url=base_url,
                timeout=120.0,  # Increase timeout for slow models
            )
        return ReflectorNode._client_cache[cache_key]
    
    async def process(self, state: SessionState) -> str:
        await self.initialize()
        user_input = state.current_input
        if not user_input:
            return ""
        # Build system prompt with memory context
        system_prompt = self.personality.generate_system_prompt()
        memory_context = ""
        if state.long_term_memory:
            # Layered context: profile + facts + summaries + semantic search
            memory_context = state.long_term_memory.build_context(
                user_input, token_budget=3000
            )
        recent = state.short_term_memory.get_context()
        messages = [{"role": "system", "content": system_prompt}]
        if memory_context:
            messages.append({"role": "system", "content": memory_context})
        messages.extend(recent)
        
        # For Qwen models with Ollama, add /no_think to disable extended reasoning
        model_name = self.config.get('llm_model', '').lower()
        user_message = user_input
        is_qwen_ollama = self.provider == "ollama" and "qwen" in model_name
        if is_qwen_ollama:
            user_message = f"{user_input} /no_think"
        if self.config.get('debug'):
            logger.debug(f"[LLM] Provider={self.provider}, Model={model_name}, Qwen+Ollama={is_qwen_ollama}")
        messages.append({"role": "user", "content": user_message})
        
        # Debug: show what we're sending (file log only)
        if self.config.get('debug'):
            total_chars = sum(len(m.get('content', '')) for m in messages)
            logger.debug(f"[LLM] Sending {len(messages)} messages, {total_chars} chars total")
        
        # Stream response & TTS
        color_print('thought', 'ReflectorNode: Thinking...')
        state.streamed_to_console = False
        full_response = ""
        try:
            async with DebugTimer(state, "llm_response"):
                async for chunk in self._stream_response(messages, state):
                    if getattr(state, "stopped", None) and state.stopped.is_set():
                        break
                    full_response += chunk
                    if self.streaming:
                        if self.tts:
                            await self.tts.stream_chunk(chunk, state)
                        # Stream text to console as chunks arrive
                        if state.config.get("stream", True):
                            import sys
                            sys.stdout.write(chunk)
                            sys.stdout.flush()
                            state.streamed_to_console = True
        except asyncio.CancelledError:
            color_print('error', 'ReflectorNode: Interrupted by wake')
            raise
        except Exception as e:
            color_print('error', f"ReflectorNode: {e}")
            full_response = "I encountered an error, sir."
        # Finalize speech (interruptible via state.stopped)
        if self.tts:
            await self.tts.finalize(state)
        # Record in short-term + long-term memory
        state.conversation_history.append({"role": "assistant", "content": full_response})
        state.short_term_memory.add("assistant", full_response)
        state.last_response = full_response
        # Persist to long-term memory (SQLite + ChromaDB + fact extraction)
        if state.long_term_memory and full_response:
            try:
                state.long_term_memory.save_interaction(user_input, full_response)
            except Exception as e:
                logger.debug(f"[Memory] save_interaction error: {e}")
        return full_response
    
    async def _stream_response(self, messages: List[Dict[str, str]], state=None):
        """Stream LLM response. Uses native Ollama API for speed, OpenAI client for other providers."""
        t_start = time.perf_counter()
        base_url = self.config.get('llm_base_url', '')
        model = self.config.get('llm_model', 'llama3.2:3b')
        
        # Use native Ollama for local models (3x faster than OpenAI compatibility layer)
        if self.provider == "ollama" or (base_url and "localhost:11434" in base_url):
            async for chunk in self._stream_ollama_native(messages, model, state, t_start):
                yield chunk
        else:
            async for chunk in self._stream_openai_client(messages, model, state, t_start):
                yield chunk
    
    async def _stream_ollama_native(self, messages: List[Dict[str, str]], model: str, state, t_start: float):
        """Stream using native Ollama API via a background thread so the event loop stays free."""
        q: queue.Queue = queue.Queue()
        _DONE = object()

        def _producer():
            try:
                import ollama
                client = ollama.Client(host="http://localhost:11434")
                logger.debug(f"[Timing] Ollama client ready: {(time.perf_counter() - t_start)*1000:.0f}ms")

                t_stream = time.perf_counter()
                stream = client.chat(
                    model=model,
                    messages=messages,
                    stream=True,
                    options={
                        "temperature": self.config.get('llm_temperature', 0.8),
                        "num_predict": self.max_tokens,
                    },
                )
                first_chunk = True
                for chunk in stream:
                    if getattr(state, "stopped", None) and state.stopped.is_set():
                        break
                    if first_chunk:
                        logger.debug(f"[Timing] Stream opened: {(t_stream - t_start)*1000:.0f}ms")
                        logger.debug(f"[Timing] First token: {(time.perf_counter() - t_stream)*1000:.0f}ms")
                        first_chunk = False
                    content = getattr(chunk.message, 'content', '') if hasattr(chunk, 'message') else ''
                    if content:
                        q.put(content)
            except Exception as exc:
                q.put(exc)
            finally:
                q.put(_DONE)

        thread = threading.Thread(target=_producer, daemon=True)
        thread.start()

        while True:
            try:
                item = await asyncio.get_event_loop().run_in_executor(None, q.get, True, 0.05)
            except Exception:          # queue.Empty on timeout
                if getattr(state, "stopped", None) and state.stopped.is_set():
                    break
                continue
            if item is _DONE:
                break
            if isinstance(item, Exception):
                color_print('error', f"Ollama streaming error: {item}")
                yield "I apologize, I encountered an error."
                break
            yield item
    
    async def _stream_openai_client(self, messages: List[Dict[str, str]], model: str, state, t_start: float):
        """Stream using OpenAI client via a background thread."""
        q: queue.Queue = queue.Queue()
        _DONE = object()

        def _producer():
            try:
                client = self._get_client()
                logger.debug(f"[Timing] OpenAI client ready: {(time.perf_counter() - t_start)*1000:.0f}ms")

                t_stream = time.perf_counter()
                stream = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=self.config.get('llm_temperature', 0.8),
                    max_tokens=self.max_tokens,
                    stream=True,
                )
                first_chunk = True
                for chunk in stream:
                    if getattr(state, "stopped", None) and state.stopped.is_set():
                        break
                    if first_chunk:
                        logger.debug(f"[Timing] First token: {(time.perf_counter() - t_stream)*1000:.0f}ms")
                        first_chunk = False
                    if chunk.choices and chunk.choices[0].delta.content:
                        q.put(chunk.choices[0].delta.content)
            except Exception as exc:
                q.put(exc)
            finally:
                q.put(_DONE)

        thread = threading.Thread(target=_producer, daemon=True)
        thread.start()

        while True:
            try:
                item = await asyncio.get_event_loop().run_in_executor(None, q.get, True, 0.05)
            except Exception:
                if getattr(state, "stopped", None) and state.stopped.is_set():
                    break
                continue
            if item is _DONE:
                break
            if isinstance(item, Exception):
                color_print('error', f"LLM streaming error: {item}")
                yield "I apologize, I encountered an error."
                break
            yield item
    


class ToolNode(Node):
    async def process(self, state: SessionState) -> Optional[str]:
        user_input = state.current_input.lower() if state.current_input else ""
        response = ""
        # Computer control
        if any(word in user_input for word in ['open', 'launch', 'start']):
            app = self._extract_app(user_input)
            if app:
                ok, msg = launch_app(app)
                response = msg
        elif 'screenshot' in user_input:
            ok, data = take_screenshot()
            if ok:
                response = "Screenshot taken."
            else:
                response = data
        elif 'status' in user_input or 'system status' in user_input:
            summary = get_system_summary()
            response = summary
        else:
            return None
        state.conversation_history.append({"role": "tool", "content": response})
        return response
    
    def _extract_app(self, text: str) -> Optional[str]:
        t = text.replace('open', '').replace('launch', '').replace('start', '').replace('my', '').replace('please', '').strip()
        t = t.strip(' ,.!?').lower()
        # First word often is the app (e.g. "calculator please" -> "calculator")
        first = t.split()[0] if t.split() else t
        common = {
            'calculator': 'calc.exe' if os.name == 'nt' else 'gnome-calculator',
            'calc': 'calc.exe' if os.name == 'nt' else 'gnome-calculator',
            'notepad': 'notepad.exe',
            'browser': 'firefox.exe' if os.name == 'nt' else 'firefox',
            'clock': 'start ms-clock:' if os.name == 'nt' else 'gnome-clocks',
            'explorer': 'explorer.exe' if os.name == 'nt' else 'nautilus',
            'file': 'explorer.exe' if os.name == 'nt' else 'nautilus',
        }
        return common.get(t, common.get(first, t or text))

class OutputNode(Node):
    async def process(self, state: SessionState) -> str:
        # If we streamed to console, only add newline (avoid double-print)
        if getattr(state, "streamed_to_console", False):
            import sys
            sys.stdout.write("\n")
            sys.stdout.flush()
        else:
            color_print('info', state.last_response or "(no response)")
        # Timers go to debug log only (file)
        if state.timers:
            for timer_name, elapsed in state.timers.items():
                logger.debug(f"[Timer] {timer_name}: {elapsed:.2f}s")
        return ""

import re

def _is_valid_time_output(text):
    return bool(re.search(r'\d{1,2}:\d{2}\s*(AM|PM|am|pm)?', text))

def time_validator_node(state, router):
    if "tool_results" in state and state["tool_results"]:
        result = state["tool_results"][0].get("result", "")
        if _is_valid_time_output(result):
            return state
    return state
class ShortTermMemory:
    def get_context(self):
        return []
    def add(self, role, content):
        pass
