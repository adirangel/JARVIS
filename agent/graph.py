"""LangGraph state machine with hybrid LLM routing.

Per-node binding:
- Planner, Reflector: DictaLM (conversation_model)
- Tool Executor: Qwen3 (tool_model) - executes tools
SQLite checkpointer for persistence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from agent.personality import JARVIS_SYSTEM_PROMPT
from agent.tools import ToolRouter, create_tool_router


def create_jarvis_graph(
    config: Optional[dict] = None,
    memory: Optional[Any] = None,
    checkpointer_path: str = "data/checkpoints",
) -> Any:
    """Build LangGraph with hybrid LLM routing.

    Flow: START -> listener -> planner -> [tool_executor if tools] -> reflector -> END
    """
    from langgraph.graph import StateGraph, END
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError:
        from langgraph_checkpoint_sqlite import SqliteSaver

    config = config or {}
    llm_cfg = config.get("llm", {})
    conv_model = llm_cfg.get("conversation_model", "aminadaven/dictalm2.0-instruct:q5_K_M")
    tool_model = llm_cfg.get("tool_model", "qwen3:4b")
    base_url = llm_cfg.get("host", "http://localhost:11434").replace("http://", "").replace("https://", "")
    if not base_url.startswith("http"):
        base_url = f"http://{base_url}"

    tool_router = create_tool_router(config)

    # State schema - use Any for messages to avoid BaseMessage scope issues with TypedDict
    from typing import TypedDict

    class AgentState(TypedDict, total=False):
        messages: list[Any]
        tool_calls: list
        tool_results: list
        final_response: str
        error: str

    # Node factories (partial application for config)
    def make_planner(state: dict) -> dict:
        from agent.nodes import planner_node
        return planner_node(
            state,
            conversation_model=conv_model,
            tool_model=tool_model,
            base_url=base_url,
            system_prompt=JARVIS_SYSTEM_PROMPT,
            tool_router=tool_router,
        )

    def make_tool_exec(state: dict) -> dict:
        from agent.nodes import tool_executor_node
        return tool_executor_node(state, tool_router=tool_router)

    def make_reflector(state: dict) -> dict:
        from agent.nodes import reflector_node
        return reflector_node(
            state,
            conversation_model=conv_model,
            base_url=base_url,
            system_prompt=JARVIS_SYSTEM_PROMPT,
        )

    # Build graph
    builder = StateGraph(AgentState)

    builder.add_node("listener", lambda s: s)  # Pass-through
    builder.add_node("planner", make_planner)
    builder.add_node("tool_executor", make_tool_exec)
    builder.add_node("reflector", make_reflector)

    builder.set_entry_point("listener")
    builder.add_edge("listener", "planner")

    def route_after_planner(state: dict) -> str:
        """If planner returned tool_calls, go to tool_executor; else reflector."""
        tool_calls = state.get("tool_calls", [])
        if tool_calls:
            return "tool_executor"
        return "reflector"

    builder.add_conditional_edges("planner", route_after_planner, {"tool_executor": "tool_executor", "reflector": "reflector"})
    builder.add_edge("tool_executor", "reflector")
    builder.add_edge("reflector", END)

    # Checkpointer - SqliteSaver for state persistence
    import sqlite3
    Path(checkpointer_path).mkdir(parents=True, exist_ok=True)
    db_path = str(Path(checkpointer_path) / "jarvis_checkpoints.db")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    graph = builder.compile(checkpointer=checkpointer)
    return graph


def _truncate_words(text: str, max_words: int = 50) -> str:
    """Limit response to max_words. Adds ... if truncated."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


def invoke_jarvis(
    graph: Any,
    user_message: str,
    thread_id: str = "default",
    stream_callback: Optional[callable] = None,
    max_words: int = 50,
) -> str:
    """Invoke JARVIS with user message. Returns final response text. Streams if stream_callback provided."""
    from langchain_core.messages import HumanMessage

    config = {"configurable": {"thread_id": thread_id}}
    from agent.nodes import _stream_callback
    token = _stream_callback.set(stream_callback) if stream_callback else None
    try:
        result = graph.invoke(
            {"messages": [HumanMessage(content=user_message)]},
            config=config,
        )
        response = result.get("final_response", "") or (
            result.get("messages", [])[-1].content if result.get("messages") else ""
        )
    finally:
        if token is not None:
            _stream_callback.reset(token)
    return _truncate_words(response, max_words)
