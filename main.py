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
        session.stopped.set()  # Signal TTS/agent to stop (do NOT clear - let running task see it)
    async with session.speech_lock:
        session.stopped.clear()  # Clear when we start our turn
        session.metadata['audio_file'] = None  # will be captured by InputNode from STT streaming
        session.current_input = transcript
        session.conversation_history.append({"role": "user", "content": transcript})
        # Run agent graph
        try:
            async for result in graph.run(session, max_turns=config.get('max_conversation_turns', 15)):
                # result from each node; output_node prints
                pass
        except asyncio.TimeoutError:
            color_print('error', "Agent error: LLM response timed out (30s)")
        except Exception as e:
            import traceback
            err_msg = str(e) or type(e).__name__
            color_print('error', f"Agent error: {err_msg}")
            if config.get('debug', False):
                traceback.print_exc()
        finally:
            # After response done, release lock after audio finishes? TTS finalize handles
            pass

async def cli_loop(session: SessionState, graph: AgentGraph, config):
    """Interactive console mode."""
    color_print('success', '[CLI] JARVIS is ready. Type your messages below. Type "exit" or "quit" to stop.')
    while True:
        try:
            user_input = input('You: ').strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in ('exit', 'quit', 'q'):
            color_print('info', '[CLI] Exiting.')
            break
        if not user_input:
            continue
        # Run a single turn via graph
        async with session.speech_lock:
            session.current_input = user_input
            session.conversation_history.append({"role": "user", "content": user_input})
            color_print('info', f'[CLI] Processing: {user_input}')
            try:
                async for result in graph.run(session, max_turns=1):
                    if result:
                        # If result is non‑empty string, it's from reflector or tool node.
                        pass
                # Response is printed by OutputNode; after graph.run finishes, we can continue.
            except Exception as e:
                color_print('error', f'Agent error: {e}')
        # Clear input for next iteration
        session.current_input = ''

async def single_text_test(session: SessionState, graph: AgentGraph, text: str):
    """One‑shot test: process a single user text and print response."""
    color_print('info', f'[Test] User: {text}')
    async with session.speech_lock:
        session.current_input = text
        session.conversation_history.append({"role": "user", "content": text})
        try:
            async for result in graph.run(session, max_turns=1):
                pass
        except Exception as e:
            color_print('error', f'Agent error: {e}')
    # Response printed by OutputNode; we can also retrieve from session.last_response
    if session.last_response:
        color_print('success', f'[Test] JARVIS: {session.last_response}')
    else:
        color_print('error', '[Test] No response generated.')

async def main():
    global config
    config = load_config()
    logger.info("Starting JARVIS with config", config=config)
    session, graph = await initialize_agent(config)
    
    if args.mode == "voice":
        # Wake listener (callback runs in worker thread - must schedule on main loop)
        wake = WakeListener(config)
        dev = config.get("input_device") or config.get("wake_device")
        dev_info = f" device={dev}" if dev is not None else " (default mic)"
        logger.info(f"Wake listener using{dev_info}. Say 'Hey Jarvis' once, then talk freely. 10 min silence = re-wake.")
        loop = asyncio.get_running_loop()
        def on_wake(transcript: str):
            asyncio.run_coroutine_threadsafe(handle_wake(transcript, session, graph, wake), loop)
        wake.start(callback=on_wake)
        logger.info("JARVIS is listening for wake word...")
        try:
            # Keep main thread alive
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            wake.stop()
    elif args.mode == "cli":
        await cli_loop(session, graph, config)
    elif args.mode == "test":
        if args.test_text:
            await single_text_test(session, graph, args.test_text)
        else:
            logger.error("--test‑text required when mode=test")
    else:
        logger.error(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="voice", choices=["voice", "cli", "test"], help="voice (wake word) or cli (console) or test (single text)")
    parser.add_argument("--test-text", help="For mode=test, provide the text input.")
    args = parser.parse_args()
    asyncio.run(main())
