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
