"""Dynamic skill loader for JARVIS self-evolution.

Scans the skills/ directory for Python modules that follow the skill contract:
- TOOL_NAME: str       — unique tool identifier
- TOOL_DESC: str       — description for the LLM
- TOOL_PARAMS: dict    — Gemini-format parameters
- execute(**kwargs) -> str  — the implementation
"""

from __future__ import annotations

import importlib.util
import os
from typing import Any, Callable

from loguru import logger

SKILLS_DIR = os.path.dirname(os.path.abspath(__file__))

REQUIRED_EXPORTS = {"TOOL_NAME", "TOOL_DESC", "TOOL_PARAMS", "execute"}

# Built-in tool names that skills cannot override
BUILTIN_TOOL_NAMES = frozenset({
    "open_app", "web_search", "weather_report", "get_current_time",
    "system_status", "computer_settings", "browser_control", "file_controller",
    "cmd_control", "code_helper", "youtube_video", "reminder", "task_manager",
    "clipboard_manager", "process_manager", "daily_briefing", "notes",
    "translate", "news", "timer", "window_manager", "screen_process",
    "computer_control", "spawn_agent", "agent_status", "agent_result",
    "stop_agent", "agent_message", "remove_agent", "skill_manager",
    "purchase_approval",
})

# Reserved filenames that are not skills
_RESERVED_FILES = {"base", "loader", "manager", "__init__", "__pycache__"}

# Registry of loaded dynamic skills: tool_name -> module
_loaded_skills: dict[str, Any] = {}

# Dispatch map: tool_name -> execute function
_skill_dispatch: dict[str, Callable] = {}


def validate_skill_module(module: Any) -> tuple[bool, str]:
    """Check that a module follows the skill contract."""
    missing = REQUIRED_EXPORTS - set(dir(module))
    if missing:
        return False, f"Missing exports: {', '.join(sorted(missing))}"
    if not isinstance(getattr(module, "TOOL_NAME", None), str):
        return False, "TOOL_NAME must be a string"
    if not isinstance(getattr(module, "TOOL_DESC", None), str):
        return False, "TOOL_DESC must be a string"
    if not isinstance(getattr(module, "TOOL_PARAMS", None), dict):
        return False, "TOOL_PARAMS must be a dict"
    if not callable(getattr(module, "execute", None)):
        return False, "execute must be callable"
    if module.TOOL_NAME in BUILTIN_TOOL_NAMES:
        return False, f"'{module.TOOL_NAME}' conflicts with a built-in tool name"
    return True, "OK"


def _make_declaration(module: Any) -> dict:
    """Convert a skill module's exports to a Gemini tool declaration."""
    params = dict(module.TOOL_PARAMS)
    if "type" not in params:
        params = {
            "type": "OBJECT",
            "properties": params.get("properties", {}),
            "required": params.get("required", []),
        }
    return {
        "name": module.TOOL_NAME,
        "description": module.TOOL_DESC,
        "parameters": params,
    }


def load_skill_from_path(file_path: str) -> tuple[bool, str, dict | None]:
    """Load a single skill from a .py file.

    Returns (success, message, tool_declaration_or_None).
    """
    if not os.path.isfile(file_path):
        return False, f"File not found: {file_path}", None

    name = os.path.splitext(os.path.basename(file_path))[0]

    if name.startswith("_") or name in _RESERVED_FILES:
        return False, f"Skipped reserved file: {name}", None

    try:
        spec = importlib.util.spec_from_file_location(f"skills.{name}", file_path)
        if spec is None or spec.loader is None:
            return False, f"Cannot create module spec for {file_path}", None

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        valid, msg = validate_skill_module(module)
        if not valid:
            return False, f"Skill '{name}' validation failed: {msg}", None

        tool_name = module.TOOL_NAME
        _loaded_skills[tool_name] = module
        _skill_dispatch[tool_name] = module.execute

        declaration = _make_declaration(module)
        logger.info(f"[Skills] Loaded skill: {tool_name} from {file_path}")
        return True, f"Loaded skill: {tool_name}", declaration

    except Exception as e:
        logger.error(f"[Skills] Failed to load {file_path}: {e}")
        return False, f"Error loading '{name}': {e}", None


def load_all_skills() -> tuple[list[dict], dict[str, Callable]]:
    """Scan skills/ directory and load all valid skill modules.

    Returns (list_of_declarations, dispatch_map).
    """
    declarations = []
    _loaded_skills.clear()
    _skill_dispatch.clear()

    if not os.path.isdir(SKILLS_DIR):
        return declarations, _skill_dispatch

    for filename in sorted(os.listdir(SKILLS_DIR)):
        if not filename.endswith(".py"):
            continue
        stem = os.path.splitext(filename)[0]
        if stem.startswith("_") or stem in _RESERVED_FILES:
            continue

        file_path = os.path.join(SKILLS_DIR, filename)
        success, msg, declaration = load_skill_from_path(file_path)
        if success and declaration:
            declarations.append(declaration)
        elif not success and "Skipped" not in msg:
            logger.warning(f"[Skills] {msg}")

    count = len(declarations)
    if count > 0:
        logger.info(f"[Skills] Loaded {count} dynamic skill(s)")

    return declarations, dict(_skill_dispatch)


def execute_skill(name: str, args: dict) -> str:
    """Execute a loaded dynamic skill by name."""
    if name not in _skill_dispatch:
        return f"Skill '{name}' not found. Use skill_manager with action 'list' to see available skills."
    try:
        return str(_skill_dispatch[name](**args))
    except Exception as e:
        logger.error(f"[Skills] Error executing {name}: {e}")
        return f"Skill error ({name}): {e}"


def get_loaded_skills() -> dict[str, Any]:
    """Return dict of currently loaded skill modules."""
    return dict(_loaded_skills)


def get_skill_dispatch() -> dict[str, Callable]:
    """Return current dispatch map."""
    return dict(_skill_dispatch)


def is_dynamic_skill(name: str) -> bool:
    """Check if a tool name belongs to a dynamic skill."""
    return name in _skill_dispatch
