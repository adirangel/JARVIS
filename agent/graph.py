"""LangGraph state machine with hybrid LLM routing.

Hybrid routing (per config.yaml):
- FastPath: Simple commands (no tools) -> direct Reflector (skips Planner)
- Planner: DictaLM - intent, planning, tool selection
- Reflector: DictaLM - final response with Paul Bettany personality
- Tool Executor: No LLM - executes tool calls from planner
- learn_new_skill: Qwen3 (tool_model) via skills_manager for code generation

SQLite checkpointer for persistence.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated, Any, Optional, TypedDict

from langgraph.graph.message import add_messages

from agent.personality import JARVIS_SYSTEM_PROMPT, PLANNER_PROMPT, REFLECTOR_PROMPT
from agent.tools import ToolRouter, create_tool_router


class AgentState(TypedDict, total=False):
    """LangGraph state schema. add_messages preserves conversation history."""
    messages: Annotated[list[Any], add_messages]
    tool_calls: list
    tool_results: list
    final_response: str
    error: str


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

    # Node factories (partial application for config)
    # Planner: DictaLM - intent, planning, tool selection
    def make_planner(state: dict) -> dict:
        from agent.nodes import planner_node
        return planner_node(
            state,
            conversation_model=conv_model,
            tool_model=tool_model,
            base_url=base_url,
            system_prompt=PLANNER_PROMPT,
            tool_router=tool_router,
            memory=memory,
            llm_config=llm_cfg,
        )

    # Tool Executor: No LLM - executes tool calls from planner
    def make_tool_exec(state: dict) -> dict:
        from agent.nodes import tool_executor_node
        return tool_executor_node(state, tool_router=tool_router)

    # Time Handler: EAGER execution - bypass planner, ALWAYS call get_current_time for time queries
    def make_time_handler(state: dict) -> dict:
        from agent.nodes import time_handler_node
        return time_handler_node(state, tool_router=tool_router)

    # Time Validator (Critic): Verify output, re-call if suspicious
    def make_time_validator(state: dict) -> dict:
        from agent.nodes import time_validator_node
        return time_validator_node(state, tool_router=tool_router)

    # Reflector: DictaLM - final response with Paul Bettany personality
    def make_reflector(state: dict) -> dict:
        from agent.nodes import reflector_node
        return reflector_node(
            state,
            conversation_model=conv_model,
            base_url=base_url,
            system_prompt=REFLECTOR_PROMPT,
            memory=memory,
            llm_config=llm_cfg,
        )

    # FastPath node: simple commands (no tools) -> direct reflector (skips planner)
    def make_fastpath(state: dict) -> dict:
        from agent.nodes import fastpath_node
        return fastpath_node(
            state,
            conversation_model=conv_model,
            base_url=base_url,
            system_prompt=REFLECTOR_PROMPT,
            memory=memory,
            llm_config=llm_cfg,
        )

    # Build graph
    builder = StateGraph(AgentState)

    builder.add_node("listener", lambda s: s)  # Pass-through
    builder.add_node("fastpath", make_fastpath)
    builder.add_node("time_handler", make_time_handler)
    builder.add_node("planner", make_planner)
    builder.add_node("tool_executor", make_tool_exec)
    builder.add_node("time_validator", make_time_validator)
    builder.add_node("reflector", make_reflector)

    builder.set_entry_point("listener")

    def route_from_listener(state: dict) -> str:
        """Time queries -> time_handler (ALWAYS use tool). Simple -> fastpath. Else -> planner."""
        messages = state.get("messages", [])
        if not messages:
            return "planner"
        last = messages[-1] if messages else None
        text = (last.content if hasattr(last, "content") else str(last)) if last else ""
        if _is_time_query(text):
            return "time_handler"  # Bypass planner - NEVER let model guess time
        # Follow-up correction ("that's wrong, check the web") - if previous user msg was time query, re-fetch
        t_lower = text.strip().lower()
        if any(p in t_lower for p in _CORRECTION_PATTERNS):
            from langchain_core.messages import HumanMessage
            for m in reversed(messages[:-1]):
                if isinstance(m, HumanMessage):
                    prev = m.content if isinstance(m.content, str) else str(m.content or "")
                    if _is_time_query(prev):
                        return "time_handler"  # Re-run with location from previous message
                    break  # Only check immediate previous user message
        if _is_simple_query(text):
            return "fastpath"
        return "planner"

    builder.add_conditional_edges(
        "listener", route_from_listener,
        {"fastpath": "fastpath", "time_handler": "time_handler", "planner": "planner"},
    )
    builder.add_edge("fastpath", END)

    def route_after_planner(state: dict) -> str:
        """If planner returned tool_calls, go to tool_executor; else reflector."""
        tool_calls = state.get("tool_calls", [])
        if tool_calls:
            return "tool_executor"
        return "reflector"

    def _route_after_tools(state: dict) -> str:
        """If we had get_current_time in results, validate; else go to reflector."""
        results = state.get("tool_results", [])
        if any(r.get("tool") == "get_current_time" for r in results):
            return "time_validator"
        return "reflector"

    builder.add_edge("time_handler", "time_validator")  # Always validate after eager time call
    builder.add_edge("time_validator", "reflector")
    builder.add_conditional_edges("planner", route_after_planner, {"tool_executor": "tool_executor", "reflector": "reflector"})
    builder.add_conditional_edges("tool_executor", _route_after_tools, {"time_validator": "time_validator", "reflector": "reflector"})
    builder.add_edge("reflector", END)

    # Checkpointer - SqliteSaver for state persistence
    import sqlite3
    Path(checkpointer_path).mkdir(parents=True, exist_ok=True)
    db_path = str(Path(checkpointer_path) / "jarvis_checkpoints.db")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    graph = builder.compile(checkpointer=checkpointer)
    return graph


def _truncate_words(text: str, max_words: int | None = 0) -> str:
    """Optional word limiter. max_words<=0 disables truncation."""
    if not text or not text.strip():
        return text
    if max_words is None or int(max_words) <= 0:
        return _ensure_complete_sentence(text)
    words = text.split()
    if len(words) <= int(max_words):
        return _ensure_complete_sentence(text)
    # Truncate at word boundary, add ellipsis
    truncated = " ".join(words[: int(max_words)]) + "..."
    return _ensure_complete_sentence(truncated)


def _ensure_complete_sentence(text: str) -> str:
    """Issue 3: If response ends without punctuation (likely cut), append witty note."""
    if not text or not text.strip():
        return text
    t = text.rstrip()
    # Proper sentence endings
    if any(t.endswith(p) for p in (".", "!", "?", "。", "…", "...")):
        return text
    # Ends mid-sentence - add note (avoids abrupt TTS cutoff)
    return text + " Pardon the interruption, Sir."


# Simple query patterns - fast path skips planner/tool routing for single LLM call
_SIMPLE_PATTERNS = (
    "hi", "hello", "hey", "thanks", "thank you", "bye", "goodbye", "ok", "okay",
    "שלום", "היי", "תודה", "בבקשה", "מה נשמע", "מה קורה", "להתראות", "יאללה",
)

# Time query patterns - ALWAYS use get_current_time tool, NEVER let model guess
_TIME_PATTERNS = (
    "what time", "current time", "time in ", "time now", "what's the time",
    "time right now", "the time", "now in ", "hora en", "quelle heure",
    "מה השעה", "שעה ב", "השעה ב", "עכשיו ב", "מה השעה ב",
)

# Correction patterns - user says previous answer was wrong
_CORRECTION_PATTERNS = ("wrong", "incorrect", "check again", "search the web", "look it up", "double-check", "that's not right")


def _is_time_query(text: str) -> bool:
    """Detect time queries - MUST use get_current_time, never fastpath or planner guess."""
    import re
    t = text.strip().lower()
    if any(p in t for p in _TIME_PATTERNS):
        return True
    # Single-word "time" at end only - "now" alone is ambiguous ("can you hear me now?" is NOT a time query)
    if re.search(r"\btime\s*\?*\s*$", t) and len(t) < 30:
        return True
    return False


def _is_simple_query(text: str) -> bool:
    """Detect greetings/short acknowledgments for fast path. Time queries go to planner."""
    if _is_time_query(text):
        return False
    t = text.strip().lower()
    if len(t) > 40:
        return False
    words = set(t.split())
    return any(p in t or p in words for p in _SIMPLE_PATTERNS)


def invoke_jarvis(
    graph: Any,
    user_message: str,
    thread_id: str = "default",
    stream_callback: Optional[callable] = None,
    max_words: int | None = 0,
    config: Optional[dict] = None,
    memory: Optional[Any] = None,
    use_fast_path: bool = True,
) -> tuple[str, float]:
    """Invoke JARVIS with user message. Returns (response_text, latency_seconds). Streams if stream_callback provided."""
    from langchain_core.messages import HumanMessage

    cfg = config or {}
    llm_cfg = cfg.get("llm", {})
    conv_model = llm_cfg.get("conversation_model", "aminadaven/dictalm2.0-instruct:q5_K_M")
    base_url = llm_cfg.get("host", "http://localhost:11434")
    if base_url and not base_url.startswith("http"):
        base_url = f"http://{base_url}"

    show_timing = cfg.get("timing", False) or cfg.get("debug", False)
    timings: list[str] = []

    def _on_timing(name: str, elapsed_ms: float) -> None:
        timings.append(f"{name}: {elapsed_ms:.0f}ms")

    # HARD BYPASS: Time queries - ALWAYS use tool, never let graph/LLM guess
    def _do_time_bypass(msg: str) -> Optional[tuple[str, float]]:
        from agent.tools import create_tool_router, extract_location_for_time_query
        from agent.tools import _BAD_LOCATION_WORDS
        loc = extract_location_for_time_query(msg)
        if not loc and memory and hasattr(memory, "get_recent_messages"):
            for m in (memory.get_recent_messages(limit=20) or []):
                prev_user = m.get("content") or m.get("user_message") or m.get("user") or ""
                if m.get("role") in ("human", "user") and prev_user and _is_time_query(prev_user):
                    loc = extract_location_for_time_query(prev_user)
                    if loc and not (set(loc.lower().split()) & _BAD_LOCATION_WORDS):
                        break
        loc = (loc or "Be'er Sheva").strip()
        try:
            router = create_tool_router(cfg)
            time_result = router.execute("get_current_time", location=loc)
            return f"The current time in {loc} is {time_result}, Sir."
        except Exception:
            return None

    if _is_time_query(user_message):
        t_start = time.perf_counter()
        result = _do_time_bypass(user_message)
        if result:
            response = result
            latency = time.perf_counter() - t_start
            if show_timing:
                print(f"[Timing] TimeBypass: {latency * 1000:.0f}ms", flush=True)
            if stream_callback:
                stream_callback(response)
            return _truncate_words(response, max_words), latency
        # Fall through to graph on error
    elif any(p in user_message.strip().lower() for p in _CORRECTION_PATTERNS) and memory:
        # "That's wrong, check the web" - try time bypass with location from last exchange
        t_start = time.perf_counter()
        result = _do_time_bypass(user_message)
        if result:
            response = result
            response = "My apologies, Sir. - " + response
            latency = time.perf_counter() - t_start
            if show_timing:
                print(f"[Timing] TimeBypass(correction): {latency * 1000:.0f}ms", flush=True)
            if stream_callback:
                stream_callback(response)
            return _truncate_words(response, max_words), latency

    # Fast path: single LLM call for simple greetings (bypasses graph - fastest)
    if use_fast_path and _is_simple_query(user_message):
        try:
            t_start = time.perf_counter()
            from langchain_ollama import ChatOllama
            from langchain_core.messages import SystemMessage
            from agent.personality import REFLECTOR_PROMPT
            mem_ctx = ""
            if memory and hasattr(memory, "build_context"):
                mem_ctx = "\n\n" + memory.build_context(user_message) if memory.build_context(user_message) else ""
            llm = ChatOllama(
                model=conv_model,
                base_url=base_url,
                temperature=0.5,
                model_kwargs={"num_predict": 128, "num_ctx": llm_cfg.get("num_ctx_reflector", 4096)},
            )
            resp = llm.invoke([
                SystemMessage(content=REFLECTOR_PROMPT + mem_ctx),
                HumanMessage(content=user_message),
            ])
            out = resp.content if hasattr(resp, "content") else str(resp)
            latency = time.perf_counter() - t_start
            if show_timing:
                print(f"[Timing] FastPath: {latency * 1000:.0f}ms", flush=True)
            return _truncate_words(out, max_words), latency
        except Exception:
            pass  # Fall through to full graph

    invoke_config = {"configurable": {"thread_id": thread_id}}
    from agent.nodes import _stream_callback, _timing_callback
    token = _stream_callback.set(stream_callback) if stream_callback else None
    timing_token = _timing_callback.set(_on_timing) if show_timing else None
    t_start = time.perf_counter()
    try:
        result = graph.invoke(
            {"messages": [HumanMessage(content=user_message)]},
            config=invoke_config,
        )
        response = result.get("final_response", "") or (
            result.get("messages", [])[-1].content if result.get("messages") else ""
        )
    finally:
        if token is not None:
            _stream_callback.reset(token)
        if timing_token is not None:
            _timing_callback.reset(timing_token)

    latency = time.perf_counter() - t_start
    if show_timing and timings:
        print(f"[Timing] {' | '.join(timings)} | Total: {latency * 1000:.0f}ms", flush=True)
    return _truncate_words(response, max_words), latency
