"""Tests for skills loading and approve_new_skill."""

import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_tool_router_loads_skills():
    """Tool router loads skills from skills/ (base.py has no TOOL_*, so no extra tools)."""
    from agent.tools import create_tool_router
    router = create_tool_router()
    assert "approve_new_skill" in router.get_tool_names()
    assert "learn_new_skill" in router.get_tool_names()


def test_approve_new_skill_no_pending():
    """approve_new_skill returns message when no pending skill."""
    from agent.tools import create_tool_router
    from skills_manager import PENDING_SKILL_FILE
    router = create_tool_router()
    if PENDING_SKILL_FILE.exists():
        PENDING_SKILL_FILE.unlink(missing_ok=True)
    result = router.execute("approve_new_skill")
    assert "No skill" in result or "pending" in result.lower() or "Sir" in result
