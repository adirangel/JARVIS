"""Conversation endpoints - GET history, POST message."""

import asyncio
import os
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.routes.uptime import increment_command_count

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
            session.long_term_memory = LongTermMemory(config)
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
    """Send a message and get JARVIS response."""
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    state = await _ensure_agent()
    session = state["session"]
    graph = state["graph"]
    config = state["config"]

    async with session.speech_lock:
        session.stopped.clear()
        session.current_input = text
        session.conversation_history.append({"role": "user", "content": text})
        increment_command_count()
        try:
            last_response = ""
            async for result in graph.run(session, max_turns=1):
                if result:
                    last_response = result
            response = getattr(session, "last_response", None) or last_response
        except Exception as e:
            response = f"I encountered an error, sir: {e}"

    return {"response": response, "messages": session.conversation_history}


@router.delete("/conversation")
async def clear_conversation():
    """Clear conversation history."""
    state = await _ensure_agent()
    session = state["session"]
    session.conversation_history = []
    if hasattr(session.short_term_memory, "_turns"):
        session.short_term_memory._turns = []
    return {"ok": True}
