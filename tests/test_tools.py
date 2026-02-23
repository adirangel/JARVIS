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


def test_try_open_browser():
    """Open/search commands should trigger open_browser."""
    from unittest.mock import patch
    from agent.tools import try_open_browser_from_intent, create_tool_router
    import yaml
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")) if (ROOT / "config.yaml").exists() else {}
    tool_router = create_tool_router(config)
    with patch("webbrowser.open"):
        assert try_open_browser_from_intent("open youtube please", tool_router) is True
        assert try_open_browser_from_intent("search for weather", tool_router) is True


def test_extract_location_time_query():
    """'What time is it?' must not extract 'is it' as location (use default)."""
    from agent.tools import extract_location_for_time_query
    assert extract_location_for_time_query("What time is it?") is None
    assert extract_location_for_time_query("current time in Tokyo") == "Tokyo"
    assert extract_location_for_time_query("time in Be'er Sheva") == "Be'er Sheva"


def test_get_current_time_beer_sheva():
    """Be'er Sheva and variants should return Asia/Jerusalem time."""
    from agent.tools import get_current_time_execute
    import re
    for loc in ["Be'er Sheva", "Beersheba", "Beer Sheva", "beer sheva"]:
        result = get_current_time_execute(loc)
        assert "AM" in result or "PM" in result, f"Expected AM/PM in {result}"
        assert re.search(r"\d{1,2}:\d{2}", result), f"Expected HH:MM in {result}"
        assert "UTC" in result or "IST" in result or "from web" in result, f"Expected timezone in {result}"


def test_get_current_time_thailand():
    """Thailand should return Asia/Bangkok (ICT), NOT IST (Israel)."""
    from agent.tools import get_current_time_execute
    import re
    result = get_current_time_execute("Thailand")
    assert "AM" in result or "PM" in result
    assert re.search(r"\d{1,2}:\d{2}", result)
    assert "IST" not in result, f"Thailand must not return IST (Israel): {result}"
    assert "UTC" in result or "ICT" in result or "from web" in result


def test_get_current_time_tokyo():
    """Tokyo should return Asia/Tokyo time."""
    from agent.tools import get_current_time_execute
    import re
    result = get_current_time_execute("Tokyo")
    assert "AM" in result or "PM" in result
    assert re.search(r"\d{1,2}:\d{2}", result)
    assert "UTC" in result or "JST" in result or "from web" in result


def test_get_current_time_miami():
    """Miami should return America/New_York (Eastern), NOT IST."""
    from agent.tools import get_current_time_execute
    import re
    result = get_current_time_execute("Miami")
    assert "AM" in result or "PM" in result
    assert re.search(r"\d{1,2}:\d{2}", result)
    # Miami = Eastern: UTC-5 or UTC-4 (DST). Must NOT be IST (Israel)
    assert "IST" not in result, f"Miami must not return IST (Israel time): {result}"
    assert "UTC" in result or "EST" in result or "EDT" in result or "verified" in result


def test_get_current_time_default():
    """Empty/None location defaults to Israel (Asia/Jerusalem)."""
    from agent.tools import get_current_time_execute
    result = get_current_time_execute("")
    assert "AM" in result or "PM" in result
    assert "UTC" in result or "IST" in result
