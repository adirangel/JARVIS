"""Skill module contract for JARVIS self-evolution.

Skills must export:
- TOOL_NAME: str - unique tool identifier
- TOOL_DESC: str - description for the LLM
- TOOL_PARAMS: dict - Ollama-style parameters {"properties": {...}, "required": [...]}
- execute(**kwargs) -> str - the implementation
"""

from __future__ import annotations

from typing import Any, Callable

# Example skill structure (for code generation prompt):
# TOOL_NAME = "my_skill"
# TOOL_DESC = "Does something useful."
# TOOL_PARAMS = {"properties": {"arg1": {"type": "string"}}, "required": ["arg1"]}
# def execute(**kwargs) -> str:
#     return "result"
