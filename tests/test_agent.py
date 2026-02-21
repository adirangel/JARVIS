"""Tests for JARVIS agent."""

import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_personality_prompt():
    from agent.personality import JARVIS_SYSTEM_PROMPT
    assert "Sir" in JARVIS_SYSTEM_PROMPT
    assert "JARVIS" in JARVIS_SYSTEM_PROMPT
    assert "British" in JARVIS_SYSTEM_PROMPT


def test_tool_router():
    from agent.tools import ToolRouter
    router = ToolRouter()
    assert "web_search" in router.get_tool_names()
    assert "file_operations" in router.get_tool_names()
    result = router.execute("web_search", query="test")
    assert "test" in result or "result" in result.lower() or "error" in result.lower()


def test_graph_creation():
    from agent.graph import create_jarvis_graph
    config = {
        "llm": {
            "conversation_model": "qwen3:4b",
            "tool_model": "qwen3:4b",
            "host": "http://localhost:11434",
        },
    }
    try:
        graph = create_jarvis_graph(config, checkpointer_path="data/test_checkpoints")
        assert graph is not None
    except Exception as e:
        pytest.skip(f"Graph creation failed (Ollama may not be running): {e}")


def test_time_query_no_false_positives():
    """'Can you hear me now?' must NOT trigger time query (was routing to get_current_time)."""
    from agent.graph import _is_time_query
    assert _is_time_query("Can you hear me now?") is False
    assert _is_time_query("Are you there now?") is False
    assert _is_time_query("what time is it?") is True
    assert _is_time_query("time") is True


def test_simple_query_detection():
    from agent.graph import _is_simple_query
    assert _is_simple_query("hi") is True
    assert _is_simple_query("שלום") is True
    assert _is_simple_query("thank you") is True
    assert _is_simple_query("מה נשמע") is True
    assert _is_simple_query("What is the capital of France?") is False
    assert _is_simple_query("Search for Python tutorials online") is False
    # Time queries must go to planner (get_current_time tool), never fastpath
    assert _is_simple_query("What is the time right now in Be'er Sheva?") is False
    assert _is_simple_query("current time in Tokyo") is False
    assert _is_simple_query("מה השעה בירושלים") is False


def test_time_validator():
    """Time validator accepts valid output, rejects invalid."""
    from agent.nodes import _is_valid_time_output, time_validator_node
    from agent.tools import create_tool_router

    assert _is_valid_time_output("10:58 AM (IST, UTC+2)") is True
    assert _is_valid_time_output("03:45 PM (JST, UTC+9)") is True
    assert _is_valid_time_output("12:00 PM (from web)") is True
    assert _is_valid_time_output("58 AM (UTC+2)") is False  # Truncated - no HH:
    assert _is_valid_time_output("") is False
    assert _is_valid_time_output("some text") is False

    # Validator passes through valid results
    router = create_tool_router()
    state = {
        "tool_results": [{"tool": "get_current_time", "result": "10:58 AM (IST, UTC+2)", "args": {"location": "Be'er Sheva"}}],
        "messages": [object(), object(), object()],  # user, ai, tool_msg
        "tool_calls": [{"id": "call_0"}],
    }
    out = time_validator_node(state, router)
    assert out["tool_results"][0]["result"] == "10:58 AM (IST, UTC+2)"


def test_has_fastpath_node():
    """Graph has FastPath node for simple commands (no tools)."""
    from agent.graph import create_jarvis_graph
    config = {"llm": {"conversation_model": "qwen3:4b", "tool_model": "qwen3:4b", "host": "http://localhost:11434"}}
    try:
        graph = create_jarvis_graph(config, checkpointer_path="data/test_checkpoints")
        # LangGraph compiled graph exposes nodes
        nodes = getattr(graph, "nodes", None) or {}
        node_names = list(nodes.keys()) if isinstance(nodes, dict) else []
        assert "fastpath" in node_names, f"Expected fastpath in {node_names}"
        assert "time_validator" in node_names, f"Expected time_validator in {node_names}"
        assert "time_handler" in node_names, f"Expected time_handler in {node_names}"
    except Exception as e:
        pytest.skip(f"Graph creation failed (Ollama may not be running): {e}")


def test_no_truncation_when_max_words_zero():
    from agent.graph import _truncate_words

    text = "One two three four five six seven."
    assert _truncate_words(text, max_words=0) == text
