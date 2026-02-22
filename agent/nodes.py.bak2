import asyncio
import os
from typing import Optional, Dict, Any, List
from agent.graph import Node, SessionState
from agent.utils import DebugTimer, color_print
from agent.personality import Personality
from audio_to_text import transcribe
from text_to_speech import TTS
from tools.computer_control import *
from tools.system_monitor import get_system_summary, format_for_speech
from memory.short_term import ShortTermMemory
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
    def __init__(self, name: str = "reflector", config: Optional[Dict[str, Any]] = None):
        super().__init__(name)
        self.config = config or {}
        self.personality = Personality(self.config.get('personality', {}))
        self.tts: Optional[TTS] = None
        self.max_tokens = self.config.get('llm_max_tokens', 500)
        self.streaming = self.config.get('streaming', True)
        self.api_key = self.config.get('llm_api_key')
        self.provider = self.config.get('llm_provider', 'stepfun')
    
    async def initialize(self):
        engine = self.config.get('tts_engine')
        if engine and engine.lower() != 'none' and self.streaming:
            self.tts = TTS(self.config)
            await self.tts.initialize()
    
    async def process(self, state: SessionState) -> str:
        await self.initialize()
        user_input = state.current_input
        if not user_input:
            return ""
        # Long-term memory
        if state.long_term_memory:
            memories = state.long_term_memory.retrieve(user_input, k=3)
            memory_context = "\n".join([doc for doc, _, _ in memories])
        else:
            memory_context = ""
        # Build messages
        system_prompt = self.personality.generate_system_prompt()
        recent = state.short_term_memory.get_context()
        messages = [{"role": "system", "content": system_prompt}]
        if memory_context:
            messages.append({"role": "system", "content": f"Relevant memories:\n{memory_context}"})
        messages.extend(recent)
        messages.append({"role": "user", "content": user_input})
        # Stream response & TTS
        color_print('thought', 'ReflectorNode: Thinking...')
        full_response = ""
        try:
            async for chunk in self._stream_response(messages):
                full_response += chunk
                if self.streaming and self.tts:
                    await self.tts.stream_chunk(chunk)
        except asyncio.CancelledError:
            color_print('error', 'ReflectorNode: Interrupted by wake')
            raise
        except Exception as e:
            color_print('error', f"ReflectorNode: {e}")
            full_response = "I encountered an error, sir."
        # Finalize speech
        if self.tts:
            await self.tts.finalize()
        # Record
        state.conversation_history.append({"role": "assistant", "content": full_response})
        state.short_term_memory.add("assistant", full_response)
        state.last_response = full_response
        self._store_important_info(user_input, full_response, state)
        return full_response
    
    async def _stream_response(self, messages: List[Dict[str, str]]):
        try:
            from openai import OpenAI
            base_url = self.config.get('llm_base_url')
            if not base_url:
                base_url = "https://api.stepfun.com/v1" if self.provider == "stepfun" else "https://openrouter.ai/api/v1"
            client = OpenAI(api_key=self.api_key, base_url=base_url)
            stream = client.chat.completions.create(
                model=self.config.get('llm_model', 'openai/gpt-4o-mini'),
                messages=messages,
                temperature=self.config.get('llm_temperature', 0.8),
                max_tokens=self.max_tokens,
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            color_print('error', f"LLM streaming error: {e}")
            yield "I apologize, I encountered an error."
    
    def _store_important_info(self, user_input: str, response: str, state: SessionState):
        if state.long_term_memory:
            if "my name is" in user_input.lower():
                words = user_input.split()
                if "is" in words:
                    idx = words.index("is")
                    if idx+1 < len(words):
                        name = words[idx+1].strip(",.!?")
                        fact = f"User's name is {name}"
                        state.long_term_memory.store(fact, category='user')
            if any(trigger in user_input.lower() for trigger in ['prefer', 'like', 'love', 'hate', 'enjoy']):
                state.long_term_memory.store(f"User preference: {user_input}", category='preference')

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
        text = text.replace('open', '').replace('launch', '').replace('start', '').strip()
        common = {
            'calculator': 'calc.exe' if os.name=='nt' else 'gnome-calculator',
            'notepad': 'notepad.exe',
            'browser': 'firefox' if os.name!='nt' else 'firefox.exe'
        }
        return common.get(text, text)

class OutputNode(Node):
    async def process(self, state: SessionState) -> str:
        color_print('info', state.last_response or "(no response)")
        if state.config.get('print_timers', True):
            for timer_name, elapsed in state.timers.items():
                color_print('info', f"[Timer] {timer_name}: {elapsed:.2f}s")
        return ""
