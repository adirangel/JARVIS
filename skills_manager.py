"""Self-evolution: learn_new_skill tool.

User: "Jarvis, learn how to control Spotify"
-> Qwen3: web search, write Python tool in ./skills/, test, ask approval, register.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Optional

# Skills directory
SKILLS_DIR = Path(__file__).parent / "skills"
PENDING_SKILL_FILE = Path(__file__).parent / "data" / "pending_skill.txt"


def _web_search(query: str, max_results: int = 5) -> str:
    try:
        from ddgs import DDGS
        results = DDGS().text(query, max_results=max_results, region="us-en")
        if not results:
            return f"No results for: {query}"
        return "\n".join(
            f"{r.get('title', '')}\n{r.get('body', '')}\n{r.get('href', '')}"
            for r in results
        )
    except Exception as e:
        return f"Search error: {e}"


def _invoke_tool_llm(prompt: str, model: str = "qwen3:4b", base_url: str = "http://localhost:11434") -> str:
    """Use Qwen3 for structured output (tool code generation)."""
    try:
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model=model, base_url=base_url, temperature=0.3)
        response = llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        return f"LLM error: {e}"


def learn_new_skill(
    skill_description: str,
    tool_router: Optional[Any] = None,
    config: Optional[dict] = None,
) -> str:
    """Implement learn_new_skill: search, generate tool, test, register.

    Returns status message for user approval.
    """
    config = config or {}
    llm_cfg = config.get("llm", {})
    tool_model = llm_cfg.get("model") or llm_cfg.get("tool_model") or llm_cfg.get("conversation_model", "qwen3:4b")
    base_url = llm_cfg.get("host", "http://localhost:11434")
    if base_url and not base_url.startswith("http"):
        base_url = f"http://{base_url}"

    SKILLS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Web search for API/docs
    search_query = f"{skill_description} API Python control"
    search_results = _web_search(search_query, max_results=5)

    # 2. Qwen3: generate Python tool (skill contract: TOOL_NAME, TOOL_DESC, TOOL_PARAMS, execute)
    prompt = f"""Create a Python tool module for: {skill_description}

Search results:
{search_results[:3000]}

Requirements - MUST export these at module level:
- TOOL_NAME: str (e.g. "spotify_control")
- TOOL_DESC: str (one line for LLM)
- TOOL_PARAMS: dict with "properties" and "required" (Ollama format)
- def execute(**kwargs) -> str: (implementation, return string result)

Use standard libraries or commonly available packages. Include error handling.
Name the file with underscores, e.g. spotify_control.py

Output ONLY the Python code, no markdown or explanation."""

    code = _invoke_tool_llm(prompt, model=tool_model, base_url=base_url)

    # Strip markdown if present
    if "```python" in code:
        code = code.split("```python", 1)[1].split("```", 1)[0].strip()
    elif "```" in code:
        code = code.split("```", 1)[1].rsplit("```", 1)[0].strip()

    # Sanitize filename
    safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in skill_description.lower())[:30]
    skill_path = SKILLS_DIR / f"{safe_name}.py"

    # 3. Write and test
    try:
        skill_path.write_text(code, encoding="utf-8")
    except Exception as e:
        return f"Error writing skill: {e}"

    # 4. Safe test - try importing
    try:
        result = subprocess.run(
            [os.environ.get("python", "python"), "-c", f"import sys; sys.path.insert(0, '{SKILLS_DIR.parent}'); exec(open('{skill_path}').read())"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(SKILLS_DIR.parent),
        )
        if result.returncode != 0:
            return f"Skill failed validation:\n{result.stderr or result.stdout}\n\nCode saved to {skill_path}. Please review and approve."
    except subprocess.TimeoutExpired:
        return f"Skill test timed out. Code saved to {skill_path}. Please review manually."

    # Store pending for approve_new_skill
    PENDING_SKILL_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        PENDING_SKILL_FILE.write_text(str(skill_path), encoding="utf-8")
    except Exception:
        pass

    return f"Skill validated: {skill_path.name}. Say 'Jarvis, approve the new skill' to register."


def approve_new_skill(tool_router: Optional[Any] = None) -> str:
    """Approve the pending skill and reload it into the tool router."""
    if not PENDING_SKILL_FILE.exists():
        return "No skill pending approval, Sir."
    try:
        path_str = PENDING_SKILL_FILE.read_text(encoding="utf-8").strip()
        skill_path = Path(path_str)
        if not skill_path.exists():
            PENDING_SKILL_FILE.unlink(missing_ok=True)
            return "Pending skill file no longer exists, Sir."
        if tool_router and hasattr(tool_router, "reload_skills"):
            tool_router.reload_skills()
        PENDING_SKILL_FILE.unlink(missing_ok=True)
        return f"Skill approved and registered: {skill_path.name}. At your service, Sir."
    except Exception as e:
        return f"I apologise, Sir. Approval failed: {e}"
