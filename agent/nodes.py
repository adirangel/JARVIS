"""LangGraph nodes - hybrid LLM routing.

- Planner: DictaLM (conversation_model) - intent, planning, tool selection
- Reflector: DictaLM (conversation_model) - final response with personality
- Tool Executor: No LLM - executes tools; learn_new_skill uses Qwen3 (tool_model) for code gen
"""

from __future__ import annotations

import contextvars
import time
from typing import Any, Optional

_stream_callback: contextvars.ContextVar[Optional[callable]] = contextvars.ContextVar("stream_callback", default=None)
_timing_callback: contextvars.ContextVar[Optional[callable]] = contextvars.ContextVar("timing_callback", default=None)


def _timing(name: str, elapsed_ms: float) -> None:
    cb = _timing_callback.get()
    if cb:
        cb(name, elapsed_ms)


# Lazy imports for optional deps
def _get_llm(
    model: str,
    base_url: str = "http://localhost:11434",
    temperature: float = 0.7,
    num_predict: Optional[int] = None,
    num_ctx: Optional[int] = None,
):
    from langchain_ollama import ChatOllama
    kwargs = {"model": model, "base_url": base_url, "temperature": temperature}
    model_kwargs = {}
    if num_predict is not None:
        model_kwargs["num_predict"] = num_predict
    if num_ctx is not None:
        model_kwargs["num_ctx"] = num_ctx
    if model_kwargs:
        kwargs["model_kwargs"] = model_kwargs
    return ChatOllama(**kwargs)


def listener_node(state: dict) -> dict:
    """Receives user input, updates state. Entry point."""
    # State already has messages from input
    return {"messages": state.get("messages", [])}


def fastpath_node(
    state: dict,
    conversation_model: str,
    base_url: str,
    system_prompt: str,
    memory: Optional[Any] = None,
    llm_config: Optional[dict] = None,
) -> dict:
    """FastPath: Simple commands (no tools) -> direct Reflector. Skips Planner for sub-800ms first token."""
    from langchain_core.messages import AIMessage, SystemMessage

    llm_cfg = llm_config or {}
    t0 = time.perf_counter()
    llm = _get_llm(
        conversation_model,
        base_url,
        temperature=llm_cfg.get("reflector_temperature", 0.5),
        num_predict=llm_cfg.get("max_tokens", 128),
        num_ctx=llm_cfg.get("num_ctx_reflector", 4096),
    )
    messages = state.get("messages", [])
    if not messages:
        return state

    last_content = ""
    for m in reversed(messages):
        if hasattr(m, "content") and m.content:
            last_content = m.content if isinstance(m.content, str) else str(m.content)
            break
    mem_ctx = _get_memory_context(memory, last_content or "general")
    personality = "CRITICAL: Stay in character. Paul Bettany JARVIS. Dry British wit. Address as Sir or אדוני."
    system = SystemMessage(content=personality + "\n\n" + system_prompt + mem_ctx)
    trimmed = _trim_messages(messages, max_turns=llm_cfg.get("context_window", 6))
    msgs = [system] + list(trimmed)
    stream_cb = _stream_callback.get()

    try:
        if stream_cb:
            chunks = []
            for chunk in llm.stream(msgs):
                if hasattr(chunk, "content") and chunk.content:
                    chunks.append(chunk.content)
                    stream_cb(chunk.content)
            content = "".join(chunks)
            response = AIMessage(content=content)
        else:
            response = llm.invoke(msgs)
            content = response.content if hasattr(response, "content") else str(response)
        elapsed = (time.perf_counter() - t0) * 1000
        _timing("FastPath", elapsed)
        return {
            "messages": list(messages) + [response],
            "final_response": content,
        }
    except Exception as e:
        return {
            "messages": messages,
            "final_response": f"I apologise, Sir. An error occurred: {e}",
        }


def _get_memory_context(memory: Any, query: str, max_turns: int = 10) -> str:
    """Build memory context (facts, recent conversations) for consistency."""
    if not memory:
        return ""
    ctx = getattr(memory, "build_context", lambda q: "")(query)
    if not ctx:
        return ""
    return f"\n\n## Context (use for consistency)\n{ctx}"


def _trim_messages(messages: list, max_turns: int = 6) -> list:
    """Keep last max_turns exchanges to avoid token overflow (shorter = faster)."""
    if len(messages) <= max_turns * 2:
        return messages
    return list(messages[-(max_turns * 2):])


def planner_node(
    state: dict,
    conversation_model: str,
    tool_model: str,
    base_url: str,
    system_prompt: str,
    tool_router: Any,
    memory: Optional[Any] = None,
    llm_config: Optional[dict] = None,
) -> dict:
    """DictaLM: Decides intent, plans response, determines if tools needed.

    Hybrid: DictaLM handles natural language understanding.
    Returns tool_calls if tools needed, else direct response plan.
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.messages import AIMessage

    llm_cfg = llm_config or {}
    t0 = time.perf_counter()
    llm = _get_llm(
        conversation_model,
        base_url,
        temperature=llm_cfg.get("planner_temperature", 0.5),
        num_predict=llm_cfg.get("planner_max_tokens", 512),
        num_ctx=llm_cfg.get("num_ctx_planner", 8192),
    )
    messages = state.get("messages", [])
    if not messages:
        return state

    # Build prompt for planning
    last_msg = messages[-1] if messages else None
    if not last_msg or not hasattr(last_msg, "content"):
        return state

    user_text = last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)
    mem_ctx = _get_memory_context(memory, user_text)
    system = SystemMessage(content=system_prompt + mem_ctx + "\n\nDecide: tools needed or direct answer? If tools: output JSON with tool names and args.")
    trimmed = _trim_messages(messages, max_turns=llm_cfg.get("context_window", 10))
    msgs = [system] + list(trimmed)

    # Bind tools for planning - Planner can request tool execution
    from langchain_core.tools import StructuredTool
    ollama_tools = tool_router.get_ollama_tools()
    lc_tools = []
    for ot in ollama_tools:
        fn = ot.get("function", {})
        name = fn.get("name", "")
        desc = fn.get("description", "Execute tool")
        if not name:
            continue
        def _make_exec(tool_name):
            def _exec(**kwargs):
                return tool_router.execute(tool_name, **kwargs)
            return _exec
        lc_tools.append(StructuredTool.from_function(
            func=_make_exec(name),
            name=name,
            description=desc,
        ))
    llm_with_tools = llm.bind_tools(lc_tools) if lc_tools else llm

    try:
        response = llm_with_tools.invoke(msgs)
        elapsed = (time.perf_counter() - t0) * 1000
        _timing("Planner", elapsed)
        tool_calls = getattr(response, "tool_calls", []) or []
        return {
            "messages": list(messages) + [response],
            "tool_calls": tool_calls,
            "planner_response": response.content if hasattr(response, "content") else str(response),
        }
    except Exception as e:
        return {
            "messages": messages,
            "tool_calls": [],
            "error": str(e),
        }


def tool_executor_node(
    state: dict,
    tool_router: Any,
) -> dict:
    """Executes tool calls (no LLM). Planner provides tool names/args; we run them.

    learn_new_skill uses Qwen3 via skills_manager._invoke_tool_llm for code generation.
    """
    t0 = time.perf_counter()
    tool_calls = state.get("tool_calls", [])
    messages = state.get("messages", [])

    if not tool_calls:
        return state

    results = []
    for tc in tool_calls:
        fn = tc.get("function", tc)
        name = fn.get("name", tc.get("name", ""))
        args = fn.get("arguments", tc.get("args", {}))
        if isinstance(args, str):
            import json
            try:
                args = json.loads(args) if args else {}
            except json.JSONDecodeError:
                args = {}
        result = tool_router.execute_tool_call({"function": {"name": name, "arguments": args}})
        results.append({"tool": name, "result": result})

    # Append tool results as message for Reflector
    from langchain_core.messages import ToolMessage
    tool_messages = []
    for i, r in enumerate(results):
        tc = tool_calls[i] if i < len(tool_calls) else {}
        tid = tc.get("id", tc.get("tool_call_id", f"call_{i}"))
        tool_messages.append(ToolMessage(content=r["result"], tool_call_id=tid))

    elapsed = (time.perf_counter() - t0) * 1000
    _timing("Tools", elapsed)
    return {
        "messages": messages + tool_messages,
        "tool_results": results,
    }


def reflector_node(
    state: dict,
    conversation_model: str,
    base_url: str,
    system_prompt: str,
    memory: Optional[Any] = None,
    llm_config: Optional[dict] = None,
) -> dict:
    """DictaLM: Formats final response with Paul Bettany personality. Streams if stream_callback in config."""
    from langchain_core.messages import AIMessage, SystemMessage

    llm_cfg = llm_config or {}
    t0 = time.perf_counter()
    llm = _get_llm(
        conversation_model,
        base_url,
        temperature=llm_cfg.get("reflector_temperature", 0.5),
        num_predict=llm_cfg.get("max_tokens", 256),
        num_ctx=llm_cfg.get("num_ctx_reflector", 4096),
    )
    messages = state.get("messages", [])

    if not messages:
        return state

    # Get last user query for memory context
    last_content = ""
    for m in reversed(messages):
        if hasattr(m, "content") and m.content:
            last_content = m.content if isinstance(m.content, str) else str(m.content)
            break
    mem_ctx = _get_memory_context(memory, last_content or "general")
    # Personality reminder: Paul Bettany JARVIS - dry British wit, address as Sir
    personality_reminder = "CRITICAL: Stay in character. Paul Bettany JARVIS. Dry British wit. Address as Sir or אדוני."
    system = SystemMessage(content=personality_reminder + "\n\n" + system_prompt + mem_ctx)
    trimmed = _trim_messages(messages, max_turns=llm_cfg.get("context_window", 10))
    msgs = [system] + list(trimmed)
    stream_cb = _stream_callback.get()

    try:
        if stream_cb:
            chunks = []
            for chunk in llm.stream(msgs):
                if hasattr(chunk, "content") and chunk.content:
                    chunks.append(chunk.content)
                    stream_cb(chunk.content)
            content = "".join(chunks)
            response = AIMessage(content=content)
        else:
            response = llm.invoke(msgs)
            content = response.content if hasattr(response, "content") else str(response)
        elapsed = (time.perf_counter() - t0) * 1000
        _timing("Reflector", elapsed)
        return {
            "messages": list(messages) + [response],
            "final_response": content,
        }
    except Exception as e:
        return {
            "messages": messages,
            "final_response": f"I apologise, Sir. An error occurred: {e}",
        }


def heartbeat_node(
    state: dict,
    memory: Any,
    conversation_model: str,
    base_url: str,
    tts_callback: Optional[callable] = None,
) -> dict:
    """Check memory for pending tasks/reminders, execute, speak witty summary."""
    # Get pending tasks and reminders
    pending = getattr(memory, "get_pending_tasks", lambda: [])()
    reminders = getattr(memory, "get_reminders", lambda: [])()

    if not pending and not reminders:
        return state

    # Build summary for DictaLM
    summary_parts = []
    if pending:
        summary_parts.append(f"Pending tasks: {len(pending)}")
    if reminders:
        summary_parts.append(f"Reminders: {len(reminders)}")

    from langchain_core.messages import HumanMessage, SystemMessage
    llm = _get_llm(conversation_model, base_url)
    prompt = f"Sir has the following: {'; '.join(summary_parts)}. Provide a brief, witty one-sentence summary to speak aloud. Stay in character as JARVIS."
    try:
        response = llm.invoke([SystemMessage(content="You are JARVIS. Brief, dry wit. Address user as Sir."), HumanMessage(content=prompt)])
        text = response.content if hasattr(response, "content") else str(response)
        if tts_callback and text:
            tts_callback(text)
    except Exception:
        pass
    return state


def self_improve_node(
    state: dict,
    tool_model: str,
    base_url: str,
    skills_manager: Any,
) -> dict:
    """Qwen3: Handles learn_new_skill - search, write tool, test, register."""
    # Delegated to skills_manager when learn_new_skill tool is invoked
    return state
