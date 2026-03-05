"""Skill management for JARVIS self-evolution.

Handles creating new skills, installing from the internet,
listing, removing, running, and reloading skills.
Also supports importing Markdown-based instruction skills (e.g. SKILL.md)
by wrapping them into Python skill modules.
"""

from __future__ import annotations

import json
import os
import re
import textwrap

from loguru import logger

from skills.loader import (
    SKILLS_DIR,
    load_skill_from_path,
    load_all_skills,
    get_loaded_skills,
    execute_skill,
)

# Patterns flagged during security review (warnings, not hard blocks)
_SECURITY_WARNINGS = [
    (r"\bos\.system\s*\(", "os.system call"),
    (r"\bsubprocess\b.*\bshell\s*=\s*True", "subprocess with shell=True"),
    (r"\beval\s*\(", "eval() call"),
    (r"\bexec\s*\(", "exec() call"),
    (r"\b__import__\s*\(", "dynamic __import__"),
    (r"\bopen\s*\(.*(/etc/|C:\\\\Windows|System32)", "system file access"),
]


def _check_security(code: str) -> list[str]:
    """Scan code for potentially dangerous patterns. Returns list of warnings."""
    warnings = []
    for pattern, desc in _SECURITY_WARNINGS:
        if re.search(pattern, code, re.IGNORECASE):
            warnings.append(desc)
    return warnings


def _sanitize_skill_name(name: str) -> str:
    """Convert a skill name to a valid Python identifier / filename."""
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9_]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def _parse_markdown_skill(content: str) -> dict | None:
    """Parse a Markdown skill file (SKILL.md format with YAML frontmatter).

    Returns dict with keys: name, description, body — or None if not a valid skill.
    """
    content = content.strip()

    # Must start with YAML frontmatter
    if not content.startswith("---"):
        return None

    # Split frontmatter from body
    parts = content.split("---", 2)
    if len(parts) < 3:
        return None

    frontmatter = parts[1].strip()
    body = parts[2].strip()

    # Parse YAML frontmatter (simple key: value)
    meta: dict[str, str] = {}
    for line in frontmatter.split("\n"):
        line = line.strip()
        if ":" in line:
            key, _, val = line.partition(":")
            meta[key.strip()] = val.strip().strip("\"'")

    name = meta.get("name", "")
    description = meta.get("description", "")

    if not name:
        return None

    return {"name": name, "description": description, "body": body}


def _convert_markdown_to_python_skill(name: str, description: str, body: str, source_url: str = "") -> str:
    """Convert a Markdown instruction skill into a Python skill module.

    The generated skill has a single 'task' parameter. When called, it returns the
    full instructions so JARVIS can follow them for the current task.
    """
    safe_name = _sanitize_skill_name(name)

    # Escape the body for embedding in a Python triple-quoted string
    escaped_body = body.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')

    source_comment = f"# Source: {source_url}\n" if source_url else ""

    return (
        f'"""JARVIS Skill: {safe_name}\n'
        f"\n"
        f"Imported from Markdown instruction skill.\n"
        f'"""\n'
        f"{source_comment}"
        f"# Installed by JARVIS skill manager (Markdown adapter)\n"
        f"\n"
        f'TOOL_NAME = "{safe_name}"\n'
        f"TOOL_DESC = {json.dumps(description or f'Workflow skill: {name}. Call this to get step-by-step instructions for this type of task.')}\n"
        f'TOOL_PARAMS = {{"type": "OBJECT", "properties": {{"task": {{"type": "STRING", "description": "Brief description of what you need to do with this skill"}}}}, "required": []}}\n'
        f"\n"
        f"_INSTRUCTIONS = \"\"\"{escaped_body}\"\"\"\n"
        f"\n"
        f"\n"
        f"def execute(**kwargs) -> str:\n"
        f'    """Return the skill instructions for the current task."""\n'
        f"    task = kwargs.get('task', '')\n"
        f"    header = f'## Skill: {safe_name}\\n'\n"
        f"    if task:\n"
        f"        header += f'Applied to: {{task}}\\n\\n'\n"
        f"    return header + _INSTRUCTIONS\n"
    )


def create_skill(
    name: str,
    description: str,
    parameters: dict,
    code: str,
) -> str:
    """Create a new skill file in the skills/ directory.

    Args:
        name: Skill name (will be sanitized to valid identifier)
        description: What the skill does (for LLM)
        parameters: Gemini-format parameter schema
        code: Python code for the execute() function body (must return a string)
    """
    safe_name = _sanitize_skill_name(name)
    if not safe_name:
        return "Invalid skill name — must contain at least one letter or digit."

    file_path = os.path.join(SKILLS_DIR, f"{safe_name}.py")
    if os.path.exists(file_path):
        return f"Skill '{safe_name}' already exists. Remove it first or choose a different name."

    # Ensure parameters have proper Gemini format
    if "type" not in parameters:
        parameters = {
            "type": "OBJECT",
            "properties": parameters.get("properties", {}),
            "required": parameters.get("required", []),
        }

    # Security scan
    warnings = _check_security(code)
    warning_comment = ""
    if warnings:
        warning_comment = f"\n# Security warnings: {', '.join(warnings)}"

    # Build the complete skill module
    # Indent the user's code body inside execute()
    indented_code = textwrap.indent(code, "    ")

    skill_content = (
        f'"""JARVIS Skill: {safe_name}\n'
        f"\n"
        f"Auto-generated by JARVIS self-evolution system.\n"
        f'"""{warning_comment}\n'
        f"\n"
        f'TOOL_NAME = "{safe_name}"\n'
        f"TOOL_DESC = {json.dumps(description)}\n"
        f"TOOL_PARAMS = {json.dumps(parameters)}\n"
        f"\n"
        f"\n"
        f"def execute(**kwargs) -> str:\n"
        f'    """Skill entry point."""\n'
        f"{indented_code}\n"
    )

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(skill_content)
    except Exception as e:
        return f"Failed to write skill file: {e}"

    # Try to load it immediately
    success, msg, declaration = load_skill_from_path(file_path)
    if not success:
        os.remove(file_path)
        return f"Skill created but failed validation: {msg}. File removed."

    result = f"Skill '{safe_name}' created and loaded successfully."
    if warnings:
        result += f" Security warnings: {', '.join(warnings)}."
    result += " Available via skill_manager 'run' action now. After restart it becomes a first-class tool."
    return result


def install_skill_from_url(url: str, skill_name: str = "") -> str:
    """Download a skill .py file from a URL and install it.

    Supports raw GitHub URLs, GitHub blob URLs (auto-converted), Gist URLs,
    direct .py file URLs, and GitHub repo URLs (auto-discovers skill files).
    """
    import httpx

    if not url:
        return "URL is required."

    # Convert GitHub blob URLs to raw URLs
    if "github.com" in url and "/blob/" in url:
        url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")

    # Detect GitHub repo URLs — crawl via API to find skill .py files
    if "github.com" in url and "/blob/" not in url and "/raw/" not in url:
        path_parts = url.rstrip("/").split("github.com/")[-1].split("/")
        # user/repo or user/repo/tree/branch — this is a repo, not a file
        if len(path_parts) <= 2 or (len(path_parts) >= 3 and path_parts[2] == "tree"):
            return _install_from_github_repo(url, path_parts)

    return _install_single_file(url, skill_name)


def _install_from_github_repo(repo_url: str, path_parts: list[str]) -> str:
    """Clone a GitHub repo (shallow), find skill files (.py or SKILL.md), install them."""
    import shutil
    import subprocess
    import tempfile

    if len(path_parts) < 2:
        return "Invalid GitHub repo URL — need at least user/repo."

    owner, repo = path_parts[0], path_parts[1]
    clone_url = f"https://github.com/{owner}/{repo}.git"

    # Shallow clone into a temp directory (no rate limits, no auth needed)
    tmp_dir = tempfile.mkdtemp(prefix="jarvis_skill_")
    try:
        logger.info(f"[Skills] Cloning {clone_url} ...")
        result = subprocess.run(
            ["git", "clone", "--depth", "1", clone_url, tmp_dir],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            return f"Failed to clone repo: {result.stderr.strip()}"

        # Walk the cloned directory to find .py skill files and SKILL.md files
        py_files: list[str] = []       # absolute paths
        md_skill_files: list[str] = [] # absolute paths

        for root, dirs, files in os.walk(tmp_dir):
            # Skip hidden dirs (.git, etc)
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for fname in files:
                fpath = os.path.join(root, fname)
                if fname.endswith(".py"):
                    py_files.append(fpath)
                elif fname == "SKILL.md":
                    md_skill_files.append(fpath)

        installed = []
        skipped = []
        failed = []
        total_scanned = len(py_files) + len(md_skill_files)

        # ── Install Python skills ──────────────────────────────────────────
        for fpath in py_files:
            fname = os.path.basename(fpath)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    code = f.read()
            except Exception as e:
                failed.append(f"{fname}: read error ({e})")
                continue

            if not re.search(r'TOOL_NAME\s*=\s*["\']', code):
                skipped.append(fname)
                continue

            has_desc = bool(re.search(r'TOOL_DESC\s*=', code))
            has_params = bool(re.search(r'TOOL_PARAMS\s*=', code))
            has_execute = bool(re.search(r'def\s+execute\s*\(', code))

            if not (has_desc and has_params and has_execute):
                skipped.append(f"{fname} (incomplete contract)")
                continue

            # Extract TOOL_NAME for the destination filename
            name_match = re.search(r'TOOL_NAME\s*=\s*["\']([^"\']+)["\']', code)
            safe_name = _sanitize_skill_name(name_match.group(1)) if name_match else _sanitize_skill_name(fname.replace(".py", ""))
            dest_path = os.path.join(SKILLS_DIR, f"{safe_name}.py")

            if os.path.exists(dest_path):
                skipped.append(f"{safe_name} (already exists)")
                continue

            # Security scan
            warnings = _check_security(code)
            source_header = f"# Source: {repo_url}\n# Installed by JARVIS skill manager\n\n"
            try:
                with open(dest_path, "w", encoding="utf-8") as f:
                    f.write(source_header + code)
            except Exception as e:
                failed.append(f"{safe_name}: write error ({e})")
                continue

            success, msg, declaration = load_skill_from_path(dest_path)
            if success:
                installed.append(safe_name)
                logger.info(f"[Skills] Installed Python skill: {safe_name}")
            else:
                os.remove(dest_path)
                failed.append(f"{safe_name}: {msg}")

        # ── Install Markdown instruction skills (SKILL.md) ────────────────
        for fpath in md_skill_files:
            rel_path = os.path.relpath(fpath, tmp_dir)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    md_content = f.read()
            except Exception as e:
                failed.append(f"{rel_path}: read error ({e})")
                continue

            parsed = _parse_markdown_skill(md_content)
            if not parsed:
                skipped.append(f"{rel_path} (no valid frontmatter)")
                continue

            skill_name = parsed["name"]
            skill_desc = parsed["description"]
            skill_body = parsed["body"]
            safe_name = _sanitize_skill_name(skill_name)
            dest_path = os.path.join(SKILLS_DIR, f"{safe_name}.py")

            if os.path.exists(dest_path):
                skipped.append(f"{safe_name} (already exists)")
                continue

            py_code = _convert_markdown_to_python_skill(
                skill_name, skill_desc, skill_body, source_url=repo_url
            )

            try:
                with open(dest_path, "w", encoding="utf-8") as f:
                    f.write(py_code)
            except Exception as e:
                failed.append(f"{safe_name}: write error ({e})")
                continue

            success, msg, declaration = load_skill_from_path(dest_path)
            if success:
                installed.append(f"{safe_name} (workflow)")
                logger.info(f"[Skills] Installed Markdown skill: {safe_name}")
            else:
                os.remove(dest_path)
                failed.append(f"{safe_name}: {msg}")

        # ── Build summary ──────────────────────────────────────────────────
        parts = [f"GitHub repo scan of {owner}/{repo}:"]
        parts.append(f"  Scanned {total_scanned} skill file(s) ({len(py_files)} .py, {len(md_skill_files)} SKILL.md)")
        if installed:
            parts.append(f"  Installed: {', '.join(installed)}")
        if skipped:
            parts.append(f"  Skipped: {', '.join(skipped[:10])}")
        if failed:
            parts.append(f"  Failed: {'; '.join(failed[:5])}")
        if not installed:
            parts.append("  No installable skills found in the repo.")
        else:
            parts.append(f"  Total: {len(installed)} skill(s) installed. Available via skill_manager 'run' or after restart as first-class tools.")

        return "\n".join(parts)
    finally:
        # Clean up the temp clone
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


def _install_single_file(url: str, skill_name: str = "") -> str:
    """Download and install a single .py skill file from a URL."""
    import httpx

    try:
        resp = httpx.get(url, timeout=30, follow_redirects=True)
        resp.raise_for_status()
        code = resp.text
    except Exception as e:
        return f"Failed to download from {url}: {e}"

    if not code.strip():
        return "Downloaded file is empty."

    # Reject HTML pages (e.g. GitHub repo pages accidentally downloaded)
    if code.strip().startswith(("<!DOCTYPE", "<html", "<!doctype")):
        return (
            "Downloaded content is an HTML page, not a Python skill file. "
            "Make sure the URL points to a raw .py file, not a web page."
        )

    # Security scan
    warnings = _check_security(code)

    # Try to determine skill name from the code
    name_match = re.search(r'TOOL_NAME\s*=\s*["\']([^"\']+)["\']', code)
    if name_match:
        detected_name = name_match.group(1)
    elif skill_name:
        detected_name = _sanitize_skill_name(skill_name)
    else:
        url_name = url.rstrip("/").split("/")[-1].replace(".py", "")
        detected_name = _sanitize_skill_name(url_name)

    if not detected_name:
        return "Cannot determine skill name. Please provide a name."

    safe_name = _sanitize_skill_name(detected_name)
    file_path = os.path.join(SKILLS_DIR, f"{safe_name}.py")

    if os.path.exists(file_path):
        return f"Skill '{safe_name}' already exists. Remove it first or choose a different name."

    # Write with source header
    source_header = f"# Source: {url}\n# Installed by JARVIS skill manager\n\n"
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(source_header + code)
    except Exception as e:
        return f"Failed to write skill file: {e}"

    # Try to load and validate
    success, msg, declaration = load_skill_from_path(file_path)
    if not success:
        os.remove(file_path)
        return f"Downloaded skill failed validation: {msg}. File removed."

    result = f"Skill '{safe_name}' installed from URL and loaded."
    if warnings:
        result += f" Security warnings: {', '.join(warnings)}."
    result += " Available via skill_manager 'run' action now. After restart it becomes a first-class tool."
    return result


def list_skills() -> str:
    """List all loaded dynamic skills."""
    skills = get_loaded_skills()
    if not skills:
        return "No dynamic skills loaded. Use 'create' to make one or 'install' to get one from the internet."

    lines = [f"Dynamic skills ({len(skills)}):"]
    for name, module in skills.items():
        desc = getattr(module, "TOOL_DESC", "No description")
        if len(desc) > 80:
            desc = desc[:77] + "..."
        lines.append(f"  - {name}: {desc}")
    return "\n".join(lines)


def remove_skill(name: str) -> str:
    """Remove a dynamic skill by name."""
    safe_name = _sanitize_skill_name(name)

    # Try both sanitized and original name
    for candidate in (safe_name, name):
        file_path = os.path.join(SKILLS_DIR, f"{candidate}.py")
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                return f"Failed to remove skill file: {e}"
            # Reload to update the registry
            load_all_skills()
            return f"Skill '{candidate}' removed successfully."

    return f"Skill file not found for '{name}'."


def run_skill(name: str, args: dict) -> str:
    """Run a dynamic skill by name with given arguments."""
    return execute_skill(name, args)


def reload_all() -> str:
    """Reload all skills from disk."""
    declarations, dispatch = load_all_skills()
    names = ", ".join(dispatch.keys()) if dispatch else "none"
    return f"Reloaded {len(declarations)} skill(s): {names}"
