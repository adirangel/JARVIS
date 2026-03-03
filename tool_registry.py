"""Tool declarations for Gemini Live API + dispatcher.

Each tool is a dict in Gemini's function_declarations format.
execute_tool() routes calls to the correct Python function.
"""

from __future__ import annotations

import json
import os
import subprocess
import platform
from typing import Any

from loguru import logger

# ── Tool declarations (Gemini format) ─────────────────────────────────────────

TOOL_DECLARATIONS = [
    {
        "name": "open_app",
        "description": "Launch any application by name or path. Examples: chrome, notepad, calc, spotify, discord, vscode",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "app_name": {
                    "type": "STRING",
                    "description": "Application name or path (e.g. 'chrome', 'notepad', 'C:\\\\Program Files\\\\...exe')"
                }
            },
            "required": ["app_name"]
        }
    },
    {
        "name": "web_search",
        "description": "Search the web for information. Returns top results with titles and snippets.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {"type": "STRING", "description": "Search query"},
                "max_results": {"type": "INTEGER", "description": "Max results (default 5)"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "weather_report",
        "description": "Get current weather for a city.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "city": {"type": "STRING", "description": "City name (e.g. 'Tel Aviv', 'London')"}
            },
            "required": ["city"]
        }
    },
    {
        "name": "get_current_time",
        "description": "Get the current date and time, optionally for a specific city/timezone.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "location": {"type": "STRING", "description": "City or timezone (e.g. 'Tokyo', 'New York'). Omit for local time."}
            },
            "required": []
        }
    },
    {
        "name": "system_status",
        "description": "Get CPU, RAM, disk usage and OS info.",
        "parameters": {
            "type": "OBJECT",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "computer_settings",
        "description": (
            "Control computer settings: volume up/down/mute, brightness, minimize/maximize/close window, "
            "switch tab, new tab, close tab, scroll up/down, zoom in/out, screenshot, lock screen, "
            "restart, shutdown, toggle dark mode, toggle wifi, copy, paste, undo, redo, "
            "select all, find, save, refresh, alt-tab, fullscreen, any keyboard shortcut. "
            "Use this for ANY single computer control command."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {
                    "type": "STRING",
                    "description": (
                        "The action: volume_up, volume_down, volume_mute, brightness_up, brightness_down, "
                        "minimize, maximize, close_window, alt_tab, new_tab, close_tab, next_tab, prev_tab, "
                        "scroll_up, scroll_down, zoom_in, zoom_out, screenshot, lock, restart, shutdown, "
                        "dark_mode, wifi_toggle, copy, paste, undo, redo, select_all, find, save, refresh, "
                        "fullscreen, escape, enter, space, hotkey"
                    )
                },
                "keys": {
                    "type": "STRING",
                    "description": "For hotkey action: key combo like 'ctrl+shift+t', 'alt+f4'"
                },
                "amount": {
                    "type": "INTEGER",
                    "description": "Repeat count (e.g. scroll_up 5 times)"
                }
            },
            "required": ["action"]
        }
    },
    {
        "name": "browser_control",
        "description": (
            "Full browser automation: navigate to URL, search, click element, type text, scroll, "
            "fill forms, read page text. Uses Playwright for reliable interaction."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {
                    "type": "STRING",
                    "description": "go_to, search, click, type, scroll, get_text, fill_form, back, forward, refresh, close"
                },
                "url": {"type": "STRING", "description": "URL for go_to action"},
                "query": {"type": "STRING", "description": "Search query for search action"},
                "selector": {"type": "STRING", "description": "CSS selector or text to find element"},
                "text": {"type": "STRING", "description": "Text to type"},
                "direction": {"type": "STRING", "description": "Scroll direction: up or down"}
            },
            "required": ["action"]
        }
    },
    {
        "name": "file_controller",
        "description": "File management: list, create, delete, move, copy, rename, read, write, find files, check disk usage.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {
                    "type": "STRING",
                    "description": "list, create, delete, move, copy, rename, read, write, find, disk_usage, mkdir"
                },
                "path": {"type": "STRING", "description": "File or directory path"},
                "destination": {"type": "STRING", "description": "Destination path (for move, copy, rename)"},
                "content": {"type": "STRING", "description": "Content to write (for write/create action)"},
                "pattern": {"type": "STRING", "description": "Search pattern (for find action)"}
            },
            "required": ["action", "path"]
        }
    },
    {
        "name": "cmd_control",
        "description": "Execute a system command (CMD/PowerShell). Use for: check disk, list processes, system info, network, etc.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "command": {"type": "STRING", "description": "The command to execute (e.g. 'ipconfig', 'tasklist', 'dir C:\\\\')"}
            },
            "required": ["command"]
        }
    },
    {
        "name": "code_helper",
        "description": "Write, edit, explain, or run code. Creates/modifies files and executes scripts.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {
                    "type": "STRING",
                    "description": "write (create file), edit (modify file), run (execute script), explain (describe code)"
                },
                "file_path": {"type": "STRING", "description": "Path to the code file"},
                "code": {"type": "STRING", "description": "Code content (for write action)"},
                "language": {"type": "STRING", "description": "Programming language (python, javascript, etc.)"},
                "explanation": {"type": "STRING", "description": "What to edit/change (for edit action)"}
            },
            "required": ["action"]
        }
    },
    {
        "name": "youtube_video",
        "description": "Play a YouTube video by search query, or get trending videos.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {
                    "type": "STRING",
                    "description": "play (search and open), trending (show trending)"
                },
                "query": {"type": "STRING", "description": "Search query for play action"}
            },
            "required": ["action"]
        }
    },
    {
        "name": "reminder",
        "description": "Set a timed reminder. Uses Windows Task Scheduler.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "message": {"type": "STRING", "description": "Reminder message"},
                "minutes": {"type": "INTEGER", "description": "Minutes from now"},
                "time": {"type": "STRING", "description": "Specific time like '14:30' or '2:30 PM'"}
            },
            "required": ["message"]
        }
    },
    {
        "name": "screen_process",
        "description": "Take a screenshot and analyze what's on screen using vision. Use when user says 'look at my screen', 'what do you see', etc.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "question": {"type": "STRING", "description": "What to look for on the screen"}
            },
            "required": []
        }
    },
    {
        "name": "computer_control",
        "description": "Direct mouse/keyboard control: click at position, type text, drag, press keys, take screenshot.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {
                    "type": "STRING",
                    "description": "click, double_click, right_click, type, press, hotkey, drag, scroll, screenshot, move_to"
                },
                "x": {"type": "INTEGER", "description": "X coordinate"},
                "y": {"type": "INTEGER", "description": "Y coordinate"},
                "text": {"type": "STRING", "description": "Text to type or key to press"},
                "keys": {"type": "STRING", "description": "Key combo for hotkey (e.g. 'ctrl+c')"},
                "end_x": {"type": "INTEGER", "description": "End X for drag"},
                "end_y": {"type": "INTEGER", "description": "End Y for drag"},
                "amount": {"type": "INTEGER", "description": "Scroll amount"}
            },
            "required": ["action"]
        }
    },
]


# ── Tool executor ─────────────────────────────────────────────────────────────

def execute_tool(name: str, args: dict) -> str:
    """Execute a tool by name. Returns result string."""
    try:
        logger.info(f"[Tool] {name}({json.dumps(args, ensure_ascii=False)[:200]})")

        if name == "open_app":
            return _exec_open_app(args)
        elif name == "web_search":
            return _exec_web_search(args)
        elif name == "weather_report":
            return _exec_weather(args)
        elif name == "get_current_time":
            return _exec_time(args)
        elif name == "system_status":
            return _exec_system_status(args)
        elif name == "computer_settings":
            return _exec_computer_settings(args)
        elif name == "browser_control":
            return _exec_browser_control(args)
        elif name == "file_controller":
            return _exec_file_controller(args)
        elif name == "cmd_control":
            return _exec_cmd(args)
        elif name == "code_helper":
            return _exec_code_helper(args)
        elif name == "youtube_video":
            return _exec_youtube(args)
        elif name == "reminder":
            return _exec_reminder(args)
        elif name == "screen_process":
            return _exec_screen_process(args)
        elif name == "computer_control":
            return _exec_computer_control(args)
        else:
            return f"Unknown tool: {name}"

    except Exception as e:
        logger.error(f"[Tool] {name} error: {e}")
        return f"Error executing {name}: {e}"


# ── Tool implementations ──────────────────────────────────────────────────────

def _exec_open_app(args: dict) -> str:
    app = args.get("app_name", "").strip()
    if not app:
        return "No application specified."
    # Common app aliases → actual commands (Windows)
    aliases = {
        "chrome": "start chrome", "google chrome": "start chrome",
        "firefox": "start firefox", "edge": "start msedge",
        "notepad": "notepad", "calc": "calc", "calculator": "calc",
        "spotify": "start spotify:", "discord": "start discord:",
        "vscode": "code", "vs code": "code", "visual studio code": "code",
        "terminal": "start wt", "powershell": "start powershell",
        "cmd": "start cmd", "explorer": "explorer",
        "paint": "mspaint", "word": "start winword",
        "excel": "start excel", "powerpoint": "start powerpnt",
        "task manager": "taskmgr", "settings": "start ms-settings:",
        "teams": "start msteams:", "outlook": "start outlook",
        "file explorer": "explorer", "control panel": "control",
        "snipping tool": "start snippingtool",
    }
    cmd = aliases.get(app.lower(), app)
    try:
        subprocess.Popen(cmd, shell=True)
        return f"Launched {app}"
    except Exception as e:
        return f"Failed to launch {app}: {e}"


def _exec_web_search(args: dict) -> str:
    query = args.get("query", "")
    max_results = args.get("max_results", 5)
    try:
        from ddgs import DDGS
        results = DDGS().text(query, max_results=max_results, region="us-en")
        if not results:
            return f"No results for: {query}"
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r.get('title', '')}: {r.get('body', '')}")
        return "\n".join(lines)
    except ImportError:
        # Fallback: open in browser
        import webbrowser, urllib.parse
        webbrowser.open(f"https://www.google.com/search?q={urllib.parse.quote_plus(query)}")
        return f"Opened Google search for: {query}"
    except Exception as e:
        return f"Search error: {e}"


def _exec_weather(args: dict) -> str:
    city = args.get("city", "Tel Aviv")
    try:
        import httpx
        # Open-Meteo geocoding + weather
        geo = httpx.get(f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1", timeout=10).json()
        if not geo.get("results"):
            return f"City not found: {city}"
        loc = geo["results"][0]
        lat, lon = loc["latitude"], loc["longitude"]
        wx = httpx.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code"
            f"&timezone=auto", timeout=10
        ).json()
        c = wx["current"]
        return (
            f"Weather in {loc['name']}: {c['temperature_2m']}°C, "
            f"humidity {c['relative_humidity_2m']}%, "
            f"wind {c['wind_speed_10m']} km/h"
        )
    except Exception as e:
        return f"Weather error: {e}"


def _exec_time(args: dict) -> str:
    from datetime import datetime
    location = args.get("location", "")
    try:
        import zoneinfo
        tz_map = {
            "london": "Europe/London", "new york": "America/New_York",
            "tokyo": "Asia/Tokyo", "paris": "Europe/Paris",
            "los angeles": "America/Los_Angeles", "sydney": "Australia/Sydney",
            "berlin": "Europe/Berlin", "moscow": "Europe/Moscow",
            "dubai": "Asia/Dubai", "singapore": "Asia/Singapore",
            "hong kong": "Asia/Hong_Kong", "bangkok": "Asia/Bangkok",
            "mumbai": "Asia/Kolkata", "israel": "Asia/Jerusalem",
            "tel aviv": "Asia/Jerusalem", "jerusalem": "Asia/Jerusalem",
            "beer sheva": "Asia/Jerusalem", "chicago": "America/Chicago",
            "miami": "America/New_York", "toronto": "America/Toronto",
        }
        tz_name = tz_map.get(location.lower().strip(), "Asia/Jerusalem") if location else "Asia/Jerusalem"
        tz = zoneinfo.ZoneInfo(tz_name)
        now = datetime.now(tz)
        return f"{now.strftime('%A, %B %d, %Y at %I:%M %p')} ({tz_name})"
    except Exception:
        now = datetime.now()
        return f"{now.strftime('%A, %B %d, %Y at %I:%M %p')} (local)"


def _exec_system_status(args: dict) -> str:
    try:
        from tools.system_monitor import get_system_summary
        return get_system_summary()
    except Exception as e:
        return f"System status error: {e}"


def _exec_computer_settings(args: dict) -> str:
    import pyautogui
    action = args.get("action", "").lower().strip()
    keys_str = args.get("keys", "")
    amount = args.get("amount", 1)

    action_map = {
        # Volume
        "volume_up": lambda: [pyautogui.press("volumeup") for _ in range(amount)],
        "volume_down": lambda: [pyautogui.press("volumedown") for _ in range(amount)],
        "volume_mute": lambda: pyautogui.press("volumemute"),
        # Window management
        "minimize": lambda: pyautogui.hotkey("win", "down"),
        "maximize": lambda: pyautogui.hotkey("win", "up"),
        "close_window": lambda: pyautogui.hotkey("alt", "F4"),
        "alt_tab": lambda: pyautogui.hotkey("alt", "tab"),
        # Tabs
        "new_tab": lambda: pyautogui.hotkey("ctrl", "t"),
        "close_tab": lambda: pyautogui.hotkey("ctrl", "w"),
        "next_tab": lambda: pyautogui.hotkey("ctrl", "tab"),
        "prev_tab": lambda: pyautogui.hotkey("ctrl", "shift", "tab"),
        # Scroll
        "scroll_up": lambda: pyautogui.scroll(amount * 3),
        "scroll_down": lambda: pyautogui.scroll(-amount * 3),
        # Zoom
        "zoom_in": lambda: [pyautogui.hotkey("ctrl", "plus") for _ in range(amount)],
        "zoom_out": lambda: [pyautogui.hotkey("ctrl", "minus") for _ in range(amount)],
        # Screenshot
        "screenshot": lambda: pyautogui.hotkey("win", "shift", "s"),
        # System
        "lock": lambda: subprocess.run("rundll32.exe user32.dll,LockWorkStation", shell=True),
        "restart": lambda: subprocess.run("shutdown /r /t 5", shell=True),
        "shutdown": lambda: subprocess.run("shutdown /s /t 5", shell=True),
        # Clipboard & editing
        "copy": lambda: pyautogui.hotkey("ctrl", "c"),
        "paste": lambda: pyautogui.hotkey("ctrl", "v"),
        "undo": lambda: pyautogui.hotkey("ctrl", "z"),
        "redo": lambda: pyautogui.hotkey("ctrl", "y"),
        "select_all": lambda: pyautogui.hotkey("ctrl", "a"),
        "find": lambda: pyautogui.hotkey("ctrl", "f"),
        "save": lambda: pyautogui.hotkey("ctrl", "s"),
        "refresh": lambda: pyautogui.press("F5"),
        "fullscreen": lambda: pyautogui.press("F11"),
        "escape": lambda: pyautogui.press("escape"),
        "enter": lambda: pyautogui.press("enter"),
        "space": lambda: pyautogui.press("space"),
        # Dark mode (Windows)
        "dark_mode": lambda: subprocess.run(
            'reg add "HKCU\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Themes\\Personalize" '
            '/v AppsUseLightTheme /t REG_DWORD /d 0 /f', shell=True
        ),
        # WiFi toggle
        "wifi_toggle": lambda: subprocess.run(
            'netsh interface set interface "Wi-Fi" disabled', shell=True
        ),
    }

    if action == "hotkey" and keys_str:
        parts = [k.strip() for k in keys_str.split("+")]
        pyautogui.hotkey(*parts)
        return f"Pressed {keys_str}"

    fn = action_map.get(action)
    if fn:
        fn()
        return f"Done: {action}"
    return f"Unknown action: {action}"


def _exec_browser_control(args: dict) -> str:
    """Browser automation via Playwright."""
    from tools.browser_control import browser_action
    return browser_action(
        action=args.get("action", ""),
        url=args.get("url", ""),
        query=args.get("query", ""),
        selector=args.get("selector", ""),
        text=args.get("text", ""),
        direction=args.get("direction", "down"),
    )


def _exec_file_controller(args: dict) -> str:
    """File operations."""
    import shutil
    action = args.get("action", "").lower()
    path = args.get("path", "")
    dest = args.get("destination", "")
    content = args.get("content", "")
    pattern = args.get("pattern", "")

    try:
        if action == "list":
            items = os.listdir(path)
            return "\n".join(items[:50]) or "(empty)"
        elif action == "read":
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()[:5000]
        elif action == "write" or action == "create":
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Written to {path}"
        elif action == "delete":
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            return f"Deleted {path}"
        elif action == "move":
            shutil.move(path, dest)
            return f"Moved {path} → {dest}"
        elif action == "copy":
            if os.path.isdir(path):
                shutil.copytree(path, dest)
            else:
                shutil.copy2(path, dest)
            return f"Copied {path} → {dest}"
        elif action == "rename":
            os.rename(path, dest)
            return f"Renamed {path} → {dest}"
        elif action == "mkdir":
            os.makedirs(path, exist_ok=True)
            return f"Created directory {path}"
        elif action == "find":
            import glob
            found = glob.glob(os.path.join(path, "**", pattern or "*"), recursive=True)
            return "\n".join(found[:30]) or "Nothing found"
        elif action == "disk_usage":
            total, used, free = shutil.disk_usage(path or "C:\\")
            return f"Total: {total // (1024**3)} GB, Used: {used // (1024**3)} GB, Free: {free // (1024**3)} GB"
        else:
            return f"Unknown file action: {action}"
    except Exception as e:
        return f"File error: {e}"


def _exec_cmd(args: dict) -> str:
    """Execute a system command."""
    command = args.get("command", "")
    if not command:
        return "No command specified."
    # Safety: block obviously dangerous commands
    dangerous = ["format", "del /s", "rd /s", "rm -rf"]
    if any(d in command.lower() for d in dangerous):
        return f"Blocked dangerous command: {command}"
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=30
        )
        output = result.stdout or result.stderr or "(no output)"
        return output[:3000]
    except subprocess.TimeoutExpired:
        return "Command timed out (30s limit)."
    except Exception as e:
        return f"Command error: {e}"


def _exec_code_helper(args: dict) -> str:
    """Code writing/editing/running."""
    action = args.get("action", "").lower()
    file_path = args.get("file_path", "")
    code = args.get("code", "")
    language = args.get("language", "python")

    if action == "write":
        if not file_path:
            return "Need file_path to write code to."
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)
        return f"Code written to {file_path}"
    elif action == "run":
        if not file_path:
            return "Need file_path to run."
        runners = {
            "python": ["python", file_path],
            "javascript": ["node", file_path],
            "typescript": ["npx", "ts-node", file_path],
        }
        cmd = runners.get(language, ["python", file_path])
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return (result.stdout or "") + (result.stderr or "") or "(no output)"
        except Exception as e:
            return f"Run error: {e}"
    elif action == "read" or action == "explain":
        if file_path and os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()[:5000]
        return "File not found."
    elif action == "edit":
        return f"To edit {file_path}: {args.get('explanation', 'No edit instructions provided')}"
    return f"Unknown code action: {action}"


def _exec_youtube(args: dict) -> str:
    """YouTube: play video or show trending."""
    action = args.get("action", "play")
    query = args.get("query", "")
    import webbrowser, urllib.parse

    if action == "play" and query:
        url = f"https://www.youtube.com/results?search_query={urllib.parse.quote_plus(query)}"
        webbrowser.open(url)
        return f"Searching YouTube for: {query}"
    elif action == "trending":
        webbrowser.open("https://www.youtube.com/feed/trending")
        return "Opened YouTube Trending."
    return "Specify action: play (with query) or trending."


def _exec_reminder(args: dict) -> str:
    """Set a reminder using Windows toast / simple approach."""
    message = args.get("message", "Reminder")
    minutes = args.get("minutes", 0)
    time_str = args.get("time", "")

    if minutes > 0:
        # Use a background thread with sleep
        import threading
        def _remind():
            import time
            time.sleep(minutes * 60)
            try:
                # Windows toast notification
                from tkinter import messagebox
                import tkinter as tk
                root = tk.Tk()
                root.withdraw()
                messagebox.showinfo("JARVIS Reminder", message)
                root.destroy()
            except Exception:
                logger.info(f"[Reminder] {message}")

        threading.Thread(target=_remind, daemon=True).start()
        return f"Reminder set for {minutes} minutes: {message}"

    return f"Reminder noted: {message}. Specify 'minutes' for a timed reminder."


def _exec_screen_process(args: dict) -> str:
    """Take a screenshot for vision analysis. Returns base64 screenshot."""
    try:
        from PIL import ImageGrab
        import base64, io
        img = ImageGrab.grab()
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        data = base64.b64encode(buf.getvalue()).decode()
        return f"SCREENSHOT_BASE64:{data}"
    except Exception as e:
        return f"Screenshot error: {e}"


def _exec_computer_control(args: dict) -> str:
    """Direct mouse/keyboard control via pyautogui."""
    import pyautogui
    action = args.get("action", "")
    x = args.get("x", 0)
    y = args.get("y", 0)
    text = args.get("text", "")
    keys = args.get("keys", "")
    amount = args.get("amount", 3)

    try:
        if action == "click":
            pyautogui.click(x, y)
        elif action == "double_click":
            pyautogui.doubleClick(x, y)
        elif action == "right_click":
            pyautogui.rightClick(x, y)
        elif action == "move_to":
            pyautogui.moveTo(x, y, duration=0.3)
        elif action == "type":
            pyautogui.write(text, interval=0.03)
        elif action == "press":
            pyautogui.press(text)
        elif action == "hotkey":
            parts = [k.strip() for k in keys.split("+")]
            pyautogui.hotkey(*parts)
        elif action == "drag":
            pyautogui.moveTo(x, y, duration=0.2)
            pyautogui.drag(args.get("end_x", 0) - x, args.get("end_y", 0) - y, duration=0.5)
        elif action == "scroll":
            pyautogui.scroll(amount)
        elif action == "screenshot":
            return _exec_screen_process({})
        else:
            return f"Unknown action: {action}"
        return f"Done: {action}"
    except Exception as e:
        return f"Control error: {e}"
