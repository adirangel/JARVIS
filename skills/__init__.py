"""JARVIS skills — dynamically loaded tools for self-evolution."""

from skills.loader import (
    load_all_skills,
    execute_skill,
    is_dynamic_skill,
    get_loaded_skills,
)

__all__ = ["load_all_skills", "execute_skill", "is_dynamic_skill", "get_loaded_skills"]
