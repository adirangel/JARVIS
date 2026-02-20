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
    tool_model = llm_cfg.get("tool_model", "qwen3:4b")
    base_url = llm_cfg.get("host", "http://localhost:11434")

    SKILLS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Web search for API/docs
    search_query = f"{skill_description} API Python control"
    search_results = _web_search(search_query, max_results=5)

    # 2. Qwen3: generate Python tool
    prompt = f"""Create a Python tool module for: {skill_description}

Search results:
{search_results[:3000]}

Requirements:
- Single file in ./skills/ with a class that has execute(**kwargs) -> str
- Use standard libraries or commonly available packages
- Return a string result
- Include error handling
- Name the file with underscores, e.g. spotify_control.py

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

    return f"Skill generated and validated: {skill_path}\n\nPlease approve to register. Say 'Jarvis, approve the new skill' to register."
