import argparse
import asyncio
import os
import sys
import yaml
from datetime import datetime
from loguru import logger

# Disable other loggers, configure loguru
logger.remove()
logger.add(
    "log/jarvis-{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="7 days",
    level="DEBUG",
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)
logger.add(sys.stderr, level="INFO", colorize=True)

from agent.graph import SessionState, AgentGraph
from agent.nodes import InputNode, ReflectorNode, ToolNode, OutputNode
from agent.personality import Personality
from agent.utils import color_print
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from voice.wake import WakeListener
from text_to_speech import TTS

def load_config():
    with open('config/settings.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    # Load LLM API key (OpenRouter, Stepfun, etc.)
    key_file = cfg.get('llm_api_key_file', 'api_key.txt')
    if os.path.exists(key_file):
        with open(key_file, 'r') as f:
            cfg['llm_api_key'] = f.read().strip()
    else:
        cfg['llm_api_key'] = None
    # Load STT API key (Whisper needs OpenAI key; falls back to llm_api_key)
    stt_key_file = cfg.get('stt_api_key_file', key_file)
    if os.path.exists(stt_key_file):
        with open(stt_key_file, 'r') as f:
            cfg['stt_api_key'] = f.read().strip()
    else:
        cfg['stt_api_key'] = cfg.get('llm_api_key')
    return cfg

async def initialize_agent(config):
    # Session state
    session = SessionState(
        session_id=f"session_{int(datetime.now().timestamp())}",
        conversation_history=[],
        timers={},
        metadata={},
        speech_lock=asyncio.Lock(),
        stopped=asyncio.Event()
    )
    session.config = config
    # Memories
    session.short_term_memory = ShortTermMemory(max_turns=config.get('max_conversation_turns', 15))
    if config.get('memory_long_term_enabled', True):
        try:
            session.long_term_memory = LongTermMemory(config)
        except Exception as e:
            logger.warning(f"Long-term memory disabled: {e}")
            session.long_term_memory = None
    else:
        session.long_term_memory = None
    # Personality
    session.personality = Personality(config.get('personality', {}))
    # Build agent graph
    input_node = InputNode(name="input")
    reflector = ReflectorNode(name="reflector", config=config)
    tool_node = ToolNode(name="tool")
    output_node = OutputNode(name="output")
    graph = AgentGraph(entry_node=input_node)
    graph.add_node(reflector)
    graph.add_node(tool_node)
    graph.add_node(output_node)
    # TTS (initialize later in reflector)
    session.tts = None
    return session, graph

async def handle_wake(transcript: str, session: SessionState, graph: AgentGraph, wake_listener: WakeListener):
    color_print('info', f"[Wake] Detected: {transcript}")
    # Debounce handled by WakeListener
    if session.speech_lock.locked():
        color_print('warn', "[Wake] Already speaking, interrupting...")
        # TODO: interrupt current TTS playback
        session.stopped.set()  # Signal nodes to stop
        await asyncio.sleep(0.1)
        session.stopped.clear()
    async with session.speech_lock:
        session.metadata['audio_file'] = None  # will be captured by InputNode from STT streaming
        session.current_input = transcript
        session.conversation_history.append({"role": "user", "content": transcript})
        # Run agent graph
        try:
            async for result in graph.run(session, max_turns=config.get('max_conversation_turns', 15)):
                # result from each node; output_node prints
                pass
        except Exception as e:
            color_print('error', f"Agent error: {e}")
        finally:
            # After response done, release lock after audio finishes? TTS finalize handles
            pass

async def main():
    global config
    config = load_config()
    logger.info("Starting JARVIS with config", config=config)
    session, graph = await initialize_agent(config)
    # Wake listener
    wake = WakeListener(config)
    # Define callback that triggers agent
    def on_wake(transcript: str):
        asyncio.create_task(handle_wake(transcript, session, graph, wake))
    wake.start(callback=on_wake)
    logger.info("JARVIS is listening for wake word...")
    try:
        # Keep main thread alive
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        wake.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="voice", choices=["voice", "cli"], help="voice (wake word) or cli")
    args = parser.parse_args()
    asyncio.run(main())
