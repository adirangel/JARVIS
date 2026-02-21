"""LangGraph nodes - hybrid LLM routing.

- Planner: DictaLM (conversation_model) - intent, planning, tool selection
- Reflector: DictaLM (conversation_model) - final response with personality
- Tool Executor: No LLM - executes tools; learn_new_skill uses Qwen3 (tool_model) for code gen
"""

from __future__ import annotations

import contextvars
from typing import Any, Optional

_stream_callback: contextvars.ContextVar[Optional[callable]] = contextvars.ContextVar("stream_callback", default=None)

# Lazy imports for optional deps
def _get_llm(model: str, base_url: str = "http://localhost:11434"):
    from langchain_ollama import ChatOllama
    return ChatOllama(model=model, base_url=base_url, temperature=0.7)


def listener_node(state: dict) -> dict:
    """Receives user input, updates state. Entry point."""
    # State already has messages from input
    return {"messages": state.get("messages", [])}


def planner_node(
    state: dict,
    conversation_model: str,
    tool_model: str,
    base_url: str,
    system_prompt: str,
    tool_router: Any,
) -> dict:
    """DictaLM: Decides intent, plans response, determines if tools needed.

    Hybrid: DictaLM handles natural language understanding.
    Returns tool_calls if tools needed, else direct response plan.
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.messages import AIMessage

    llm = _get_llm(conversation_model, base_url)
    messages = state.get("messages", [])
    if not messages:
        return state

    # Build prompt for planning
    last_msg = messages[-1] if messages else None
    if not last_msg or not hasattr(last_msg, "content"):
        return state

    user_text = last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)
    system = SystemMessage(content=system_prompt + "\n\nDecide: tools needed or direct answer? If tools: output JSON with tool names and args.")
    msgs = [system] + list(messages)

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

    return {
        "messages": messages + tool_messages,
        "tool_results": results,
    }


def reflector_node(
    state: dict,
    conversation_model: str,
    base_url: str,
    system_prompt: str,
) -> dict:
    """DictaLM: Formats final response with Paul Bettany personality. Streams if stream_callback in config."""
    from langchain_core.messages import AIMessage, SystemMessage

    llm = _get_llm(conversation_model, base_url)
    messages = state.get("messages", [])

    if not messages:
        return state

    # Personality reminder: Paul Bettany JARVIS - dry British wit, address as Sir
    personality_reminder = "CRITICAL: Stay in character. Paul Bettany JARVIS. Dry British wit. Address as Sir or אדוני."
    system = SystemMessage(content=personality_reminder + "\n\n" + system_prompt)
    msgs = [system] + list(messages)
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
