"""Conversation endpoints - GET history, POST message (with WS token streaming)."""

import asyncio
import os
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.routes.uptime import increment_command_count
from api.websocket import broadcast_conversation, broadcast_token

router = APIRouter(prefix="/api", tags=["conversation"])

# Lazy-initialized agent (session, graph, config)
_agent_state = None


def _get_project_root() -> Path:
    """Project root (parent of api/)."""
    return Path(__file__).resolve().parent.parent.parent


def _load_config():
    """Load config from config/settings.yaml."""
    import yaml
    root = _get_project_root()
    cfg_path = root / "config" / "settings.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    key_file = cfg.get("llm_api_key_file", "api_key.txt")
    key_path = root / key_file
    if key_path.exists():
        with open(key_path, "r") as f:
            cfg["llm_api_key"] = f.read().strip()
    else:
        cfg["llm_api_key"] = None
    cfg["stt_api_key"] = cfg.get("llm_api_key")
    return cfg


async def _ensure_agent():
    """Initialize agent on first use."""
    global _agent_state
    if _agent_state is not None:
        return _agent_state
    os.chdir(_get_project_root())
    config = _load_config()
    from agent.graph import SessionState, AgentGraph
    from agent.nodes import InputNode, ReflectorNode, ToolNode, OutputNode
    from memory.short_term import ShortTermMemory
    from memory.long_term import LongTermMemory
    from agent.personality import Personality

    session = SessionState(
        session_id=f"session_{int(datetime.now().timestamp())}",
        conversation_history=[],
        timers={},
        metadata={},
        speech_lock=asyncio.Lock(),
        stopped=asyncio.Event(),
    )
    session.config = config
    session.short_term_memory = ShortTermMemory(
        max_turns=config.get("max_conversation_turns", 15)
    )
    if config.get("memory_long_term_enabled", True):
        try:
            from main import _make_llm_fn
            llm_fn = _make_llm_fn(config)
            mem_cfg = {
                "db_path": config.get("memory_db_path", "data/jarvis.db"),
                "chroma_path": config.get("memory_chroma_path", "data/chroma"),
                "embedding_model": config.get("memory_embeddings_model", "nomic-embed-text"),
                "ollama_host": config.get("memory_ollama_host", "http://localhost:11434"),
            }
            session.long_term_memory = LongTermMemory({"memory": mem_cfg}, llm_fn=llm_fn)
        except Exception:
            session.long_term_memory = None
    else:
        session.long_term_memory = None
    session.personality = Personality(config.get("personality", {}))
    session.tts = None

    input_node = InputNode(name="input")
    reflector = ReflectorNode(name="reflector", config=config)
    tool_node = ToolNode(name="tool")
    output_node = OutputNode(name="output")
    graph = AgentGraph(entry_node=input_node)
    graph.add_node(reflector)
    graph.add_node(tool_node)
    graph.add_node(output_node)

    _agent_state = {"session": session, "graph": graph, "config": config}
    return _agent_state


class SendMessageRequest(BaseModel):
    text: str


WELCOME_MESSAGE = {
    "role": "assistant",
    "content": "Hello, I am JARVIS. How can I assist you today, sir?",
}


@router.get("/conversation")
async def get_conversation():
    """Return conversation history."""
    state = await _ensure_agent()
    messages = state["session"].conversation_history
    if not messages:
        return {"messages": [WELCOME_MESSAGE]}
    return {"messages": messages}


@router.post("/conversation")
async def post_conversation(req: SendMessageRequest):
    """Send a message and get JARVIS response.  Streams tokens to WS clients in real-time."""
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    state = await _ensure_agent()
    session = state["session"]
    graph = state["graph"]

    async with session.speech_lock:
        session.stopped.clear()
        session.current_input = text
        session.conversation_history.append({"role": "user", "content": text})
        increment_command_count()

        # Broadcast user message immediately
        await broadcast_conversation(session.conversation_history)

        try:
            last_response = ""
            async for result in graph.run(session, max_turns=1):
                if result:
                    last_response = result
            response = getattr(session, "last_response", None) or last_response
        except Exception as e:
            response = f"I encountered an error, sir: {e}"

    # Broadcast full conversation with assistant response
    await broadcast_conversation(session.conversation_history)

    return {"response": response, "messages": session.conversation_history}


@router.post("/conversation/stream")
async def post_conversation_stream(req: SendMessageRequest):
    """Send a message and stream response as Server-Sent Events (SSE).

    Each SSE event is one of:
      data: {"token": "chunk"}      — partial token
      data: {"done": true, "response": "full text"}  — final
    """
    import json

    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    state = await _ensure_agent()
    session = state["session"]

    async def _generate():
        from agent.nodes import ReflectorNode
        graph = state["graph"]

        async with session.speech_lock:
            session.stopped.clear()
            session.current_input = text
            session.conversation_history.append({"role": "user", "content": text})
            increment_command_count()
            await broadcast_conversation(session.conversation_history)

            # Find the ReflectorNode so we can stream its output
            reflector = None
            for node in [graph._entry_node] + list(getattr(graph, '_nodes', {}).values()):
                if isinstance(node, ReflectorNode):
                    reflector = node
                    break

            full_response = ""
            if reflector:
                await reflector.initialize()
                # Build messages exactly as ReflectorNode.process() does
                system_prompt = reflector.personality.generate_system_prompt()
                memory_context = ""
                if session.long_term_memory:
                    memory_context = session.long_term_memory.build_context(text, token_budget=3000)
                recent = session.short_term_memory.get_context()
                messages = [{"role": "system", "content": system_prompt}]
                if memory_context:
                    messages.append({"role": "system", "content": memory_context})
                messages.extend(recent)

                model_name = reflector.config.get('llm_model', '').lower()
                user_message = text
                if reflector.provider == "ollama" and "qwen" in model_name:
                    user_message = f"{text} /no_think"
                messages.append({"role": "user", "content": user_message})

                async for chunk in reflector._stream_response(messages, session):
                    if getattr(session, "stopped", None) and session.stopped.is_set():
                        break
                    full_response += chunk
                    # Push to WebSocket clients AND SSE
                    await broadcast_token(chunk)
                    yield f"data: {json.dumps({'token': chunk})}\n\n"
            else:
                # Fallback: non-streaming
                try:
                    async for result in graph.run(session, max_turns=1):
                        if result:
                            full_response = result
                    full_response = getattr(session, "last_response", None) or full_response
                except Exception as e:
                    full_response = f"I encountered an error, sir: {e}"

            # Finalize
            session.conversation_history.append({"role": "assistant", "content": full_response})
            session.short_term_memory.add("assistant", full_response)
            session.last_response = full_response
            if session.long_term_memory and full_response:
                try:
                    session.long_term_memory.save_interaction(text, full_response)
                except Exception:
                    pass
            await broadcast_token("", done=True)
            await broadcast_conversation(session.conversation_history)
            yield f"data: {json.dumps({'done': True, 'response': full_response})}\n\n"

        session.current_input = ""

    return StreamingResponse(_generate(), media_type="text/event-stream")


@router.delete("/conversation")
async def clear_conversation():
    """Clear conversation history."""
    state = await _ensure_agent()
    session = state["session"]
    session.conversation_history = []
    if hasattr(session.short_term_memory, "_turns"):
        session.short_term_memory._turns = []
    return {"ok": True}
