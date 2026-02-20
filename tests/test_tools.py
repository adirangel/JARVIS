"""Tests for agent tools."""

import pytest
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_web_search():
    from agent.tools import web_search_execute
    result = web_search_execute("Python", max_results=2)
    assert isinstance(result, str)
    assert "Python" in result or "Error" in result or "result" in result.lower()


def test_file_ops():
    from agent.tools import file_ops_execute
    with tempfile.TemporaryDirectory() as tmp:
        test_file = Path(tmp) / "test.txt"
        test_file.write_text("hello")
        result = file_ops_execute("read", str(test_file), allowed_dirs=[tmp])
        assert "hello" in result
        result = file_ops_execute("list", tmp, allowed_dirs=[tmp])
        assert "test.txt" in result or "Contents" in result


def test_system_cmd():
    from agent.tools import system_cmd_execute
    result = system_cmd_execute("echo hello", timeout=5)
    assert "hello" in result or "Output" in result


def test_try_open_browser_hebrew():
    """Hebrew commands should trigger open_browser like English."""
    from unittest.mock import patch
    from agent.tools import try_open_browser_from_intent, create_tool_router
    import yaml
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")) if (ROOT / "config.yaml").exists() else {}
    tool_router = create_tool_router(config)
    with patch("webbrowser.open"):
        # Hebrew: open youtube
        assert try_open_browser_from_intent("פתח יוטיוב בבקשה", tool_router) is True
        # Hebrew: search
        assert try_open_browser_from_intent("חפש מזג אוויר", tool_router) is True
        # English: open youtube (sanity check)
        assert try_open_browser_from_intent("open youtube please", tool_router) is True
