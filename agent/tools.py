"""Dynamic tool loader - base tools + skills/ directory.

Tools are registered for LangGraph/Ollama function calling.
Hybrid LLM: Tool Executor node uses Qwen3 for tool calls.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Optional

# Base tool interface for Ollama
def _to_ollama_tool(name: str, description: str, parameters: dict) -> dict:
    """Convert to Ollama tool format."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": parameters.get("properties", {}), "required": parameters.get("required", [])},
        },
    }


# --- Open Browser / URL ---
KNOWN_SITES = {
    "youtube": "https://youtube.com",
    "google": "https://google.com",
    "gmail": "https://gmail.com",
    "github": "https://github.com",
    "twitter": "https://twitter.com",
    "x": "https://x.com",
    "facebook": "https://facebook.com",
    "reddit": "https://reddit.com",
    "wikipedia": "https://wikipedia.org",
    "netflix": "https://netflix.com",
    "spotify": "https://spotify.com",
    "grok": "https://grok.com",
}
# Hebrew names for same sites (for "פתח יוטיוב" etc.)
KNOWN_SITES_HEBREW = {
    "יוטיוב": "https://youtube.com",
    "גוגל": "https://google.com",
    "גימייל": "https://gmail.com",
    "גיטהאב": "https://github.com",
    "טוויטר": "https://twitter.com",
    "פייסבוק": "https://facebook.com",
    "רדיט": "https://reddit.com",
    "ויקיפדיה": "https://wikipedia.org",
    "נטפליקס": "https://netflix.com",
    "ספוטיפיי": "https://spotify.com",
    "גרוק": "https://grok.com",  # Grok
}


def open_browser_execute(url: str = "", search_query: str = "") -> str:
    """Open a URL or search in the default browser (Chrome if set as default)."""
    try:
        import urllib.parse
        import webbrowser
        if search_query:
            q = urllib.parse.quote_plus(search_query.strip())
            target = f"https://www.google.com/search?q={q}"
        elif url and url.strip():
            target = url.strip()
            if not target.startswith(("http://", "https://")):
                target = "https://" + target
        else:
            return "Error: No URL or search query provided."
        webbrowser.open(target)
        return f"Opened {target} in browser."
    except Exception as e:
        return f"Error opening browser: {str(e)}"


def try_open_browser_from_intent(user_text: str, tool_router: Any) -> bool:
    """If user asks to open/search something, open browser immediately. Returns True if opened."""
    import re
    text_raw = user_text.strip()
    text_lower = text_raw.lower()
    # Trigger words: English + Hebrew (פתח=open, חפש/חיפוש=search, דפדפן=browser, עבור/לך=go to)
    triggers_en = ("open", "search", "go to", "navigate", "browser")
    triggers_he = ("פתח", "חפש", "חיפוש", "דפדפן", "עבור", "לך", "תפתח", "תחפש")
    if not any(w in text_lower for w in triggers_en) and not any(w in text_raw for w in triggers_he):
        return False
    # Search: "search for X", "search X", "חפש X", "חיפוש X", "תחפש X"
    m = re.search(
        r"(?:search\s+(?:for\s+)?|חפש\s+|חיפוש\s+|תחפש\s+)(.+?)(?:\.|$|please|בבקשה|תודה)",
        text_raw, re.I | re.UNICODE
    )
    if m:
        q = m.group(1).strip()
        if q and len(q) > 1:
            tool_router.execute("open_browser", search_query=q)
            return True
    # Open site: English names
    for site, url in KNOWN_SITES.items():
        if site in text_lower:
            tool_router.execute("open_browser", url=url)
            return True
    # Open site: Hebrew names (יוטיוב, גוגל, etc.)
    for site_he, url in KNOWN_SITES_HEBREW.items():
        if site_he in text_raw:
            tool_router.execute("open_browser", url=url)
            return True
    # "open browser" / "פתח דפדפן" / "פתח לי דפדפן" -> open Google
    if "דפדפן" in text_raw and ("פתח" in text_raw or "תפתח" in text_raw or "open" in text_lower):
        tool_router.execute("open_browser", url="https://google.com")
        return True
    if "browser" in text_lower and ("open" in text_lower or "פתח" in text_raw):
        tool_router.execute("open_browser", url="https://google.com")
        return True
    # Open X: "פתח X", "תפתח X", "תפתח לי X" - extract X after trigger
    m = re.search(
        r"(?:open|פתח|תפתח|go\s+to|עבור\s+ל|לך\s+ל)\s*(?:לי\s+)?(.+?)(?:\.|$|בבקשה|please|תודה)",
        text_raw, re.I | re.UNICODE
    )
    if m:
        target = m.group(1).strip()
        if target and len(target) > 1:
            # Check if it's a known Hebrew site
            for site_he, url in KNOWN_SITES_HEBREW.items():
                if site_he in target:
                    tool_router.execute("open_browser", url=url)
                    return True
            # Check if it's a known English site
            for site, url in KNOWN_SITES.items():
                if site in target.lower():
                    tool_router.execute("open_browser", url=url)
                    return True
            # Otherwise treat as search query
            tool_router.execute("open_browser", search_query=target)
            return True
    # Open URL-like: "open https://...", "go to example.com"
    m = re.search(r"(?:open|go\s+to|navigate\s+to)\s+([a-zA-Z0-9][-a-zA-Z0-9.]*\.[a-zA-Z]{2,}(?:\S*)?)", text_lower)
    if m:
        u = m.group(1).strip()
        if u and len(u) > 3:
            tool_router.execute("open_browser", url=u)
            return True
    return False


OPEN_BROWSER_TOOL = _to_ollama_tool(
    "open_browser",
    "Open URL or search in browser. Use for: open youtube, פתח יוטיוב, search for X, חפש X, open browser, פתח דפדפן. ALWAYS use when user asks to open or search (Hebrew or English).",
    {
        "properties": {
            "url": {"type": "string", "description": "URL to open (e.g. https://youtube.com)"},
            "search_query": {"type": "string", "description": "Search query - opens Google search"},
        },
        "required": [],
    },
)


# --- Web Search (ddgs / duckduckgo-search) ---
def web_search_execute(query: str, max_results: int = 5) -> str:
    try:
        from ddgs import DDGS
        results = DDGS().text(query, max_results=max_results, region="us-en")
        if not results:
            return f"No results found for: {query}"
        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(f"{i}. {r.get('title', 'No title')}\n   {r.get('body', '')}\n   URL: {r.get('href', '')}")
        return f"Search results for '{query}':\n\n" + "\n\n".join(formatted)
    except ImportError:
        return "Error: ddgs not installed. pip install ddgs"
    except Exception as e:
        return f"Search error: {str(e)}"


WEB_SEARCH_TOOL = _to_ollama_tool(
    "web_search",
    "Search the web using DuckDuckGo. Use for information lookup.",
    {"properties": {"query": {"type": "string", "description": "Search query"}, "max_results": {"type": "integer", "description": "Max results", "default": 5}}, "required": ["query"]},
)


# --- Current Time (accurate, uses system clock + timezone) ---
_CITY_TO_TZ = {
    "beer sheva": "Asia/Jerusalem",
    "beer sheba": "Asia/Jerusalem",
    "be'er sheva": "Asia/Jerusalem",
    "beersheba": "Asia/Jerusalem",
    "jerusalem": "Asia/Jerusalem",
    "tel aviv": "Asia/Jerusalem",
    "tel aviv-yafo": "Asia/Jerusalem",
    "haifa": "Asia/Jerusalem",
    "london": "Europe/London",
    "new york": "America/New_York",
    "los angeles": "America/Los_Angeles",
    "chicago": "America/Chicago",
    "paris": "Europe/Paris",
    "berlin": "Europe/Berlin",
    "tokyo": "Asia/Tokyo",
    "sydney": "Australia/Sydney",
}


def get_current_time_execute(location: str = "") -> str:
    """Get the current date and time. Uses system clock - always accurate. Location is optional (city name for timezone)."""
    from datetime import datetime
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        try:
            from backports.zoneinfo import ZoneInfo
        except ImportError:
            ZoneInfo = None
    if ZoneInfo is None:
        from datetime import timezone
        return datetime.now(timezone.utc).strftime("%A, %B %d, %Y at %I:%M:%S %p UTC")
    tz_name = "UTC"
    if location and location.strip():
        loc_lower = location.strip().lower().replace("'", "").replace("'", "")
        tz_name = _CITY_TO_TZ.get(loc_lower)
        if not tz_name:
            # Try partial match
            for city, tz in _CITY_TO_TZ.items():
                if city in loc_lower or loc_lower in city:
                    tz_name = tz
                    break
            else:
                tz_name = "UTC"
        try:
            tz = ZoneInfo(tz_name)
        except Exception:
            tz = ZoneInfo("UTC")
    else:
        tz = ZoneInfo("UTC")
    now = datetime.now(tz)
    return now.strftime("%A, %B %d, %Y at %I:%M:%S %p %Z")


CURRENT_TIME_TOOL = _to_ollama_tool(
    "get_current_time",
    "Get the ACCURATE current date and time. ALWAYS use this for 'what time is it', 'current time', 'time in X' - never guess. Uses system clock. Location: city name (e.g. Be'er Sheva, London, New York) for timezone.",
    {"properties": {"location": {"type": "string", "description": "City/location for timezone (e.g. Be'er Sheva, London). Empty = UTC."}}, "required": []},
)


# --- File Operations ---
def file_ops_execute(action: str, path: str, content: str = "", allowed_dirs: Optional[list[str]] = None) -> str:
    path_obj = Path(os.path.expanduser(path)).resolve()
    if allowed_dirs:
        allowed = [Path(os.path.expanduser(d)).resolve() for d in allowed_dirs]
        if not any(path_obj == a or a in path_obj.parents for a in allowed):
            return f"Error: Access to {path} not allowed."
    try:
        if action == "read":
            if not path_obj.exists():
                return f"Error: {path} does not exist."
            if not path_obj.is_file():
                return f"Error: {path} is not a file."
            text = path_obj.read_text(encoding="utf-8")
            if len(text) > 10000:
                text = text[:10000] + f"\n\n... (truncated, total {len(text)} chars)"
            return f"File content:\n\n{text}"
        elif action == "write":
            if not content:
                return "Error: No content for write."
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            path_obj.write_text(content, encoding="utf-8")
            return f"Written successfully ({len(content)} chars)."
        elif action == "list":
            if not path_obj.exists():
                return f"Error: {path} does not exist."
            if not path_obj.is_dir():
                return f"Error: {path} is not a directory."
            items = sorted(path_obj.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
            lines = [f"Contents of {path_obj}:"]
            for item in items[:50]:
                prefix = "[DIR]" if item.is_dir() else "[FILE]"
                lines.append(f"  {prefix} {item.name}")
            return "\n".join(lines)
        elif action == "delete":
            if path_obj.is_file():
                path_obj.unlink()
                return "Deleted successfully."
            return "Error: Cannot delete directories."
        else:
            return f"Error: Unknown action {action}"
    except Exception as e:
        return f"Error: {str(e)}"


FILE_OPS_TOOL = _to_ollama_tool(
    "file_operations",
    "Read, write, list, delete files. Use for file management.",
    {
        "properties": {
            "action": {"type": "string", "enum": ["read", "write", "list", "delete"], "description": "Action"},
            "path": {"type": "string", "description": "File or directory path"},
            "content": {"type": "string", "description": "Content for write"},
        },
        "required": ["action", "path"],
    },
)


# --- System Command ---
BLOCKED_CMDS = {"format", "del /s", "rd /s", "rmdir /s", "rm -rf /", "shutdown", "restart"}


def system_cmd_execute(command: str, working_directory: Optional[str] = None, timeout: int = 30) -> str:
    cmd_lower = command.lower().strip()
    for blocked in BLOCKED_CMDS:
        if blocked in cmd_lower:
            return f"Error: Command blocked for safety."
    try:
        import subprocess
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=working_directory,
            encoding="utf-8",
            errors="replace",
        )
        parts = []
        if result.stdout:
            out = result.stdout.strip()
            if len(out) > 5000:
                out = out[:5000] + "\n... (truncated)"
            parts.append(f"Output:\n{out}")
        if result.stderr:
            err = result.stderr.strip()
            if len(err) > 2000:
                err = err[:2000] + "\n... (truncated)"
            parts.append(f"Errors:\n{err}")
        parts.append(f"Exit code: {result.returncode}")
        return "\n\n".join(parts)
    except subprocess.TimeoutExpired:
        return f"Error: Command exceeded {timeout}s timeout."
    except Exception as e:
        return f"Error: {str(e)}"


SYSTEM_CMD_TOOL = _to_ollama_tool(
    "system_command",
    "Run system commands (CMD/PowerShell). Use with caution.",
    {
        "properties": {
            "command": {"type": "string", "description": "Command to run"},
            "working_directory": {"type": "string", "description": "Working directory"},
        },
        "required": ["command"],
    },
)


# --- Computer Control (pyautogui) ---
def computer_control_execute(action: str, **kwargs) -> str:
    try:
        import pyautogui
        if action == "screenshot":
            path = kwargs.get("path", "screenshot.png")
            pyautogui.screenshot().save(path)
            return f"Screenshot saved to {path}"
        elif action == "click":
            x, y = kwargs.get("x", 0), kwargs.get("y", 0)
            pyautogui.click(x, y)
            return f"Clicked at ({x}, {y})"
        elif action == "type":
            text = kwargs.get("text", "")
            pyautogui.write(text, interval=0.05)
            return f"Typed {len(text)} chars"
        elif action == "hotkey":
            keys = kwargs.get("keys", "ctrl+c")
            pyautogui.hotkey(*keys.split("+"))
            return f"Pressed {keys}"
        else:
            return f"Error: Unknown action {action}"
    except ImportError:
        return "Error: pyautogui not installed."
    except Exception as e:
        return f"Error: {str(e)}"


# --- Learn New Skill (self-evolution, uses Qwen3 via skills_manager) ---
LEARN_NEW_SKILL_TOOL = _to_ollama_tool(
    "learn_new_skill",
    "When user asks to learn a new skill (e.g. control Spotify): search web, write Python tool in ./skills/, test, ask for approval.",
    {"properties": {"skill_description": {"type": "string", "description": "What to learn, e.g. 'control Spotify'"}}, "required": ["skill_description"]},
)

# --- Approve New Skill (register pending skill from learn_new_skill) ---
APPROVE_NEW_SKILL_TOOL = _to_ollama_tool(
    "approve_new_skill",
    "When user says 'approve the new skill' or 'אשר את המיומנות': register the pending skill from learn_new_skill.",
    {"properties": {}, "required": []},
)


COMPUTER_CONTROL_TOOL = _to_ollama_tool(
    "computer_control",
    "Control mouse, keyboard, take screenshots. Use for UI automation.",
    {
        "properties": {
            "action": {"type": "string", "enum": ["screenshot", "click", "type", "hotkey"]},
            "path": {"type": "string"},
            "x": {"type": "integer"},
            "y": {"type": "integer"},
            "text": {"type": "string"},
            "keys": {"type": "string"},
        },
        "required": ["action"],
    },
)


# --- Tool Router ---
class ToolRouter:
    """Execute tools by name. Used by Tool Executor node."""

    def __init__(self, config: Optional[dict] = None):
        self._config = config or {}
        self._tools: dict[str, Callable[..., str]] = {}
        self._ollama_tools: list[dict] = []
        self._skill_tool_names: set[str] = set()
        self._register_defaults()

    def _register_defaults(self) -> None:
        tools_config = self._config.get("tools", {})
        allowed = tools_config.get("allowed_directories", ["~"])
        timeout = tools_config.get("command_timeout", 30)
        max_results = tools_config.get("max_search_results", 5)

        def _file_ops(**kw):
            return file_ops_execute(
                kw.get("action", ""), kw.get("path", ""), kw.get("content", ""), allowed
            )

        def _sys_cmd(**kw):
            return system_cmd_execute(
                kw.get("command", ""), kw.get("working_directory"), timeout
            )

        def _open_browser(**kw):
            return open_browser_execute(kw.get("url", ""), kw.get("search_query", ""))
        self.register("open_browser", _open_browser, OPEN_BROWSER_TOOL)
        self.register("web_search", lambda **kw: web_search_execute(kw.get("query", ""), kw.get("max_results", max_results)), WEB_SEARCH_TOOL)
        self.register("get_current_time", lambda **kw: get_current_time_execute(kw.get("location", "")), CURRENT_TIME_TOOL)
        self.register("file_operations", _file_ops, FILE_OPS_TOOL)
        self.register("system_command", _sys_cmd, SYSTEM_CMD_TOOL)
        self.register("computer_control", lambda **kw: computer_control_execute(kw.get("action", ""), **kw), COMPUTER_CONTROL_TOOL)

        def _learn_skill(**kw):
            from skills_manager import learn_new_skill
            return learn_new_skill(
                kw.get("skill_description", ""),
                tool_router=self,
                config=self._config,
            )
        self.register("learn_new_skill", _learn_skill, LEARN_NEW_SKILL_TOOL)

        def _approve_skill(**kw):
            from skills_manager import approve_new_skill
            return approve_new_skill(tool_router=self)
        self.register("approve_new_skill", _approve_skill, APPROVE_NEW_SKILL_TOOL)

    def register(self, name: str, execute_fn: Callable[..., str], ollama_spec: dict) -> None:
        self._tools[name] = execute_fn
        # Avoid duplicate ollama specs
        self._ollama_tools = [t for t in self._ollama_tools if t["function"]["name"] != name]
        self._ollama_tools.append(ollama_spec)

    def execute(self, name: str, **kwargs) -> str:
        fn = self._tools.get(name)
        if not fn:
            return f"Error: Unknown tool {name}"
        try:
            return fn(**kwargs)
        except Exception as e:
            return f"Error: {str(e)}"

    def execute_tool_call(self, tool_call: dict) -> str:
        fn = tool_call.get("function", {})
        name = fn.get("name", "")
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args) if args else {}
            except json.JSONDecodeError:
                args = {}
        return self.execute(name, **args)

    def get_ollama_tools(self) -> list[dict]:
        return self._ollama_tools.copy()

    def get_tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def _load_skills(self) -> None:
        """Load skill modules from ./skills/ and register them."""
        import importlib.util
        import sys
        skills_dir = Path(__file__).parent.parent / "skills"
        if not skills_dir.exists():
            return
        root = str(skills_dir.parent)
        if root not in sys.path:
            sys.path.insert(0, root)
        for f in sorted(skills_dir.glob("*.py")):
            if f.name.startswith("_") or f.name == "base.py":
                continue
            try:
                spec = importlib.util.spec_from_file_location(f"skill_{f.stem}", f)
                if spec is None or spec.loader is None:
                    continue
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                name = getattr(mod, "TOOL_NAME", None)
                desc = getattr(mod, "TOOL_DESC", "Skill")
                params = getattr(mod, "TOOL_PARAMS", {"properties": {}, "required": []})
                execute_fn = getattr(mod, "execute", None)
                if name and callable(execute_fn):
                    ollama_spec = _to_ollama_tool(name, desc, params)
                    self.register(name, execute_fn, ollama_spec)
                    self._skill_tool_names.add(name)
            except Exception:
                pass

    def reload_skills(self) -> None:
        """Remove skill tools and reload from ./skills/."""
        for name in list(self._skill_tool_names):
            self._tools.pop(name, None)
            self._ollama_tools = [t for t in self._ollama_tools if t.get("function", {}).get("name") != name]
        self._skill_tool_names.clear()
        self._load_skills()


def create_tool_router(config: Optional[dict] = None) -> ToolRouter:
    """Create tool router with default + dynamically loaded skills."""
    router = ToolRouter(config)
    router._load_skills()
    return router
