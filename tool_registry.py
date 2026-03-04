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
            "Control the user's real Chrome/Edge browser: navigate to URL, Google search, "
            "type text in any field or chat, click, scroll, open/close tabs, go back/forward, "
            "focus address bar, find text on page. Uses the actual browser, not a separate instance. "
            "Use type_and_enter to send messages in chat interfaces."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {
                    "type": "STRING",
                    "description": (
                        "go_to, search, click, type, type_and_enter, scroll, new_tab, close_tab, "
                        "next_tab, prev_tab, back, forward, refresh, address, find, "
                        "select_all, copy, paste, close, focus"
                    )
                },
                "url": {"type": "STRING", "description": "URL for go_to/new_tab/address actions"},
                "query": {"type": "STRING", "description": "Search query for search action"},
                "selector": {"type": "STRING", "description": "For click: coordinates as 'x,y'"},
                "text": {"type": "STRING", "description": "Text to type (supports Hebrew/Unicode)"},
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
        "description": (
            "Execute a system command (CMD/PowerShell). Use for: check disk, list processes, system info, network, etc. "
            "Set visible=true to run in a new visible terminal window the user can see."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "command": {"type": "STRING", "description": "The command to execute (e.g. 'ipconfig', 'tasklist', 'dir C:\\\\')"},
                "visible": {"type": "BOOLEAN", "description": "If true, runs in a new visible terminal window. Default false."}
            },
            "required": ["command"]
        }
    },
    {
        "name": "code_helper",
        "description": (
            "Write, edit, explain, or run code. Creates/modifies files and executes scripts. "
            "'run' executes in a visible terminal so user sees the output. "
            "'run_background' opens a persistent terminal window for long-running scripts."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {
                    "type": "STRING",
                    "description": (
                        "write (create file), edit (modify file), run (execute and show output), "
                        "run_background (run in visible terminal that stays open), "
                        "read (read file), explain (describe code)"
                    )
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
        "description": (
            "Manage reminders: set a new reminder, list all active reminders, or mark one as done. "
            "Reminders persist across sessions. Use 'list' to show all, 'set' to create, 'done' to complete."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {
                    "type": "STRING",
                    "description": "set (create reminder), list (show all active), done (mark completed)"
                },
                "message": {"type": "STRING", "description": "Reminder message (for set action)"},
                "minutes": {"type": "INTEGER", "description": "Minutes from now (for set action)"},
                "time": {"type": "STRING", "description": "Specific time like '14:30' or '2:30 PM' (for set action). Can also be 'tomorrow 14:30'"},
                "reminder_id": {"type": "INTEGER", "description": "Reminder ID (for done action)"}
            },
            "required": ["action"]
        }
    },
    {
        "name": "task_manager",
        "description": (
            "Manage tasks/to-do list: add new tasks, list pending, mark complete, delete. "
            "Tasks persist across sessions. The user can ask to remember things to do."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {
                    "type": "STRING",
                    "description": "add, list, complete, delete"
                },
                "description": {"type": "STRING", "description": "Task description (for add)"},
                "due": {"type": "STRING", "description": "Optional due date/time like 'tomorrow 14:00' or '2026-03-10'"},
                "task_id": {"type": "INTEGER", "description": "Task ID (for complete/delete)"}
            },
            "required": ["action"]
        }
    },
    {
        "name": "clipboard_manager",
        "description": "Read from or write to the system clipboard.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "read or write"},
                "text": {"type": "STRING", "description": "Text to copy to clipboard (for write)"}
            },
            "required": ["action"]
        }
    },
    {
        "name": "process_manager",
        "description": "List running processes, kill a process, or check if a specific app is running.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "list, kill, check"},
                "name": {"type": "STRING", "description": "Process name to kill or check (e.g. 'chrome', 'notepad')"},
                "sort_by": {"type": "STRING", "description": "Sort list by: cpu, memory, name (default: memory)"}
            },
            "required": ["action"]
        }
    },
    {
        "name": "daily_briefing",
        "description": (
            "Generate a daily briefing for the user: current time, weather, system status, "
            "pending tasks, upcoming reminders, and top news headlines. The morning report."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "city": {"type": "STRING", "description": "City for weather (default: Tel Aviv)"}
            },
            "required": []
        }
    },
    {
        "name": "notes",
        "description": "Quick persistent notes: save, search, list, or delete notes. Uses the JARVIS knowledge base.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "save, list, search, delete"},
                "title": {"type": "STRING", "description": "Note title or key (for save/delete)"},
                "content": {"type": "STRING", "description": "Note content (for save)"},
                "query": {"type": "STRING", "description": "Search query (for search)"}
            },
            "required": ["action"]
        }
    },
    {
        "name": "translate",
        "description": "Translate text between languages. Supports Hebrew, English, Spanish, French, German, Arabic, Russian, Chinese, Japanese, and more.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "text": {"type": "STRING", "description": "Text to translate"},
                "target_language": {"type": "STRING", "description": "Target language code or name (e.g. 'he', 'en', 'es', 'hebrew', 'english')"},
                "source_language": {"type": "STRING", "description": "Source language (auto-detect if omitted)"}
            },
            "required": ["text", "target_language"]
        }
    },
    {
        "name": "news",
        "description": "Get current news headlines. Optionally filter by topic or country.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "topic": {"type": "STRING", "description": "News topic query (e.g. 'technology', 'sports', 'Israel')"},
                "max_results": {"type": "INTEGER", "description": "Number of headlines (default 5)"}
            },
            "required": []
        }
    },
    {
        "name": "timer",
        "description": "Set a countdown timer. When done, shows a notification. Supports minutes or seconds.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "minutes": {"type": "INTEGER", "description": "Timer duration in minutes"},
                "seconds": {"type": "INTEGER", "description": "Timer duration in seconds (alternative to minutes)"},
                "label": {"type": "STRING", "description": "Timer label (e.g. 'Pasta timer')"}
            },
            "required": []
        }
    },
    {
        "name": "window_manager",
        "description": "List open windows, switch to a window, snap/arrange windows, or minimize all.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {
                    "type": "STRING",
                    "description": "list, switch, snap_left, snap_right, minimize_all, restore_all"
                },
                "window_name": {"type": "STRING", "description": "Window title to switch to or snap (partial match)"}
            },
            "required": ["action"]
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
    # ── Agent management tools ────────────────────────────────────────────────
    {
        "name": "spawn_agent",
        "description": (
            "Spawn a new background agent to work on a task independently. "
            "The agent runs in the background and reports results back. "
            "Use this when the user asks for long-running tasks, parallel work, "
            "monitoring, research, or anything that should run independently. "
            "Types: 'command' (run a command), 'script' (run a file), "
            "'monitor' (watch something periodically), 'research' (web search), "
            "'tool' (use a JARVIS tool), 'multi_step' (sequential steps), 'general' (auto-detect)."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "name": {
                    "type": "STRING",
                    "description": "Agent name (e.g. 'researcher', 'cpu-monitor', 'code-runner')"
                },
                "task": {
                    "type": "STRING",
                    "description": "Description of what the agent should do"
                },
                "task_type": {
                    "type": "STRING",
                    "description": "Task type: command, script, monitor, research, tool, multi_step, general"
                },
                "command": {
                    "type": "STRING",
                    "description": "Command to execute (for command/monitor types)"
                },
                "file_path": {
                    "type": "STRING",
                    "description": "Script file to run (for script type)"
                },
                "query": {
                    "type": "STRING",
                    "description": "Search query (for research type)"
                },
                "tool_name": {
                    "type": "STRING",
                    "description": "Tool to execute (for tool type)"
                },
                "tool_args": {
                    "type": "STRING",
                    "description": "JSON string of tool arguments (for tool type)"
                },
                "interval_seconds": {
                    "type": "INTEGER",
                    "description": "Check interval in seconds (for monitor type, default 10)"
                },
                "visible": {
                    "type": "BOOLEAN",
                    "description": "If true, run script in a visible terminal window"
                }
            },
            "required": ["name", "task"]
        }
    },
    {
        "name": "agent_status",
        "description": "Check the status of spawned agents. Shows all agents if no name specified, or detailed info for one agent.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "name": {
                    "type": "STRING",
                    "description": "Agent name to check (omit for all agents)"
                }
            },
            "required": []
        }
    },
    {
        "name": "agent_result",
        "description": "Get the full result/output of a completed agent.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "name": {
                    "type": "STRING",
                    "description": "Agent name to get result from"
                }
            },
            "required": ["name"]
        }
    },
    {
        "name": "stop_agent",
        "description": "Stop a running agent.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "name": {
                    "type": "STRING",
                    "description": "Agent name to stop"
                }
            },
            "required": ["name"]
        }
    },
    {
        "name": "agent_message",
        "description": (
            "Send a message between agents. Agents can communicate with each other. "
            "Use recipient 'all' to broadcast to all agents."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "sender": {
                    "type": "STRING",
                    "description": "Sender name (agent name or 'jarvis')"
                },
                "recipient": {
                    "type": "STRING",
                    "description": "Recipient agent name, or 'all' for broadcast"
                },
                "message": {
                    "type": "STRING",
                    "description": "Message content"
                }
            },
            "required": ["recipient", "message"]
        }
    },
    {
        "name": "remove_agent",
        "description": "Remove a stopped or completed agent from the registry.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "name": {
                    "type": "STRING",
                    "description": "Agent name to remove"
                }
            },
            "required": ["name"]
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
        elif name == "task_manager":
            return _exec_task_manager(args)
        elif name == "clipboard_manager":
            return _exec_clipboard(args)
        elif name == "process_manager":
            return _exec_process_manager(args)
        elif name == "daily_briefing":
            return _exec_daily_briefing(args)
        elif name == "notes":
            return _exec_notes(args)
        elif name == "translate":
            return _exec_translate(args)
        elif name == "news":
            return _exec_news(args)
        elif name == "timer":
            return _exec_timer(args)
        elif name == "window_manager":
            return _exec_window_manager(args)
        elif name == "screen_process":
            return _exec_screen_process(args)
        elif name == "computer_control":
            return _exec_computer_control(args)
        # Agent management
        elif name == "spawn_agent":
            return _exec_spawn_agent(args)
        elif name == "agent_status":
            return _exec_agent_status(args)
        elif name == "agent_result":
            return _exec_agent_result(args)
        elif name == "stop_agent":
            return _exec_stop_agent(args)
        elif name == "agent_message":
            return _exec_agent_message(args)
        elif name == "remove_agent":
            return _exec_remove_agent(args)
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
    visible = args.get("visible", False)
    if not command:
        return "No command specified."
    # Safety: block obviously dangerous commands
    dangerous = ["format", "del /s", "rd /s", "rm -rf"]
    if any(d in command.lower() for d in dangerous):
        return f"Blocked dangerous command: {command}"

    if visible and platform.system() == "Windows":
        # Run in a visible terminal window
        try:
            full_cmd = f'start "JARVIS Command" cmd /k "{command}"'
            subprocess.Popen(full_cmd, shell=True)
            return f"Running in visible terminal: {command}"
        except Exception as e:
            return f"Failed to open visible terminal: {e}"

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
        return _run_script_visible(file_path, language, background=False)
    elif action == "run_background":
        if not file_path:
            return "Need file_path to run."
        return _run_script_visible(file_path, language, background=True)
    elif action == "read" or action == "explain":
        if file_path and os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()[:5000]
        return "File not found."
    elif action == "edit":
        return f"To edit {file_path}: {args.get('explanation', 'No edit instructions provided')}"
    return f"Unknown code action: {action}"


def _run_script_visible(file_path: str, language: str = "python", background: bool = False) -> str:
    """Run a script in a visible terminal window so the user can see output.

    If background=False, waits for completion and captures output.
    If background=True, opens a persistent terminal that stays open.
    """
    if not os.path.exists(file_path):
        return f"File not found: {file_path}"

    abs_path = os.path.abspath(file_path)
    work_dir = os.path.dirname(abs_path)

    runners = {
        "python": f'python "{abs_path}"',
        "javascript": f'node "{abs_path}"',
        "typescript": f'npx ts-node "{abs_path}"',
        "batch": f'"{abs_path}"',
        "powershell": f'powershell -File "{abs_path}"',
    }
    # Auto-detect language from extension if not specified
    ext = os.path.splitext(file_path)[1].lower()
    ext_map = {".py": "python", ".js": "javascript", ".ts": "typescript",
               ".bat": "batch", ".cmd": "batch", ".ps1": "powershell"}
    if language == "python" and ext in ext_map:
        language = ext_map[ext]

    run_cmd = runners.get(language, f'python "{abs_path}"')

    if platform.system() == "Windows":
        if background:
            # Open a new visible console that stays open
            full_cmd = f'start "JARVIS Script" cmd /k "cd /d {work_dir} && {run_cmd}"'
            subprocess.Popen(full_cmd, shell=True)
            return f"Launched in visible terminal: {file_path}\nThe script is running in a new window."
        else:
            # Run in a new visible console, capture output, window stays for 3s then closes
            # But ALSO capture output back to JARVIS
            try:
                result = subprocess.run(
                    run_cmd, shell=True, capture_output=True, text=True,
                    timeout=60, cwd=work_dir
                )
                output = (result.stdout or "") + (result.stderr or "")
                output = output.strip()

                # Also show in a visible terminal if there's meaningful output
                if output:
                    # Open notepad-style popup or terminal with output
                    _show_script_output(file_path, output)

                return output or "(script completed with no output)"
            except subprocess.TimeoutExpired:
                return "Script timed out after 60 seconds. Use run_background for long-running scripts."
            except Exception as e:
                return f"Run error: {e}"
    else:
        # Linux/Mac
        try:
            result = subprocess.run(
                run_cmd, shell=True, capture_output=True, text=True,
                timeout=60, cwd=work_dir
            )
            output = (result.stdout or "") + (result.stderr or "")
            return output.strip() or "(script completed with no output)"
        except Exception as e:
            return f"Run error: {e}"


def _show_script_output(file_path: str, output: str):
    """Show script output in a visible window so the user can see it."""
    try:
        # Create a temp file with the output and show it
        import tempfile
        out_file = os.path.join(tempfile.gettempdir(), "jarvis_script_output.txt")
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(f"=== JARVIS Script Output: {file_path} ===\n")
            f.write(f"{'=' * 50}\n\n")
            f.write(output)
            f.write(f"\n\n{'=' * 50}\n")
            f.write("Script finished.\n")
        # Open in default text viewer
        if platform.system() == "Windows":
            subprocess.Popen(["notepad", out_file])
    except Exception as e:
        logger.warning(f"Could not show script output: {e}")


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


def _parse_future_time(time_str: str, minutes: int = 0) -> float:
    """Parse a time string into a future unix timestamp."""
    import re
    from datetime import datetime, timedelta

    now = datetime.now()

    if minutes and minutes > 0:
        return (now + timedelta(minutes=minutes)).timestamp()

    if not time_str:
        return (now + timedelta(minutes=5)).timestamp()

    ts = time_str.strip().lower()

    # "tomorrow 14:30" or "tomorrow 2:30 pm"
    tomorrow = False
    if "tomorrow" in ts:
        tomorrow = True
        ts = ts.replace("tomorrow", "").strip()

    # Try HH:MM or H:MM (24h)
    m = re.match(r"^(\d{1,2}):(\d{2})$", ts)
    if m:
        h, mi = int(m.group(1)), int(m.group(2))
        target = now.replace(hour=h, minute=mi, second=0, microsecond=0)
        if tomorrow:
            target += timedelta(days=1)
        elif target <= now:
            target += timedelta(days=1)
        return target.timestamp()

    # "2:30 PM" / "2:30PM"
    m = re.match(r"^(\d{1,2}):(\d{2})\s*(am|pm)$", ts)
    if m:
        h, mi = int(m.group(1)), int(m.group(2))
        if m.group(3) == "pm" and h != 12:
            h += 12
        elif m.group(3) == "am" and h == 12:
            h = 0
        target = now.replace(hour=h, minute=mi, second=0, microsecond=0)
        if tomorrow:
            target += timedelta(days=1)
        elif target <= now:
            target += timedelta(days=1)
        return target.timestamp()

    # "in 30 minutes" / "in 2 hours"
    m = re.match(r"in\s+(\d+)\s+(minute|min|hour|hr|second|sec)s?", ts)
    if m:
        val = int(m.group(1))
        unit = m.group(2)
        if "hour" in unit or "hr" in unit:
            return (now + timedelta(hours=val)).timestamp()
        elif "sec" in unit:
            return (now + timedelta(seconds=val)).timestamp()
        else:
            return (now + timedelta(minutes=val)).timestamp()

    # Fallback: 5 minutes from now
    return (now + timedelta(minutes=5)).timestamp()


def _get_sqlite_store():
    """Get a shared SQLiteStore instance."""
    from memory.sqlite_store import SQLiteStore
    return SQLiteStore("data/jarvis.db")


def _exec_reminder(args: dict) -> str:
    """Manage reminders: set, list, done — persisted in SQLite."""
    from datetime import datetime
    action = args.get("action", "set").lower()
    message = args.get("message", "")
    minutes = args.get("minutes", 0)
    time_str = args.get("time", "")
    rem_id = args.get("reminder_id", 0)

    try:
        store = _get_sqlite_store()

        if action == "list":
            # Get ALL active (not done) reminders
            rows = store._conn.execute(
                "SELECT id, text, trigger_at FROM reminders WHERE done = 0 ORDER BY trigger_at"
            ).fetchall()
            if not rows:
                return "No active reminders."
            lines = []
            now = datetime.now()
            for r in rows:
                rid = r["id"]
                text = r["text"]
                trigger = r["trigger_at"]
                try:
                    dt = datetime.fromtimestamp(trigger)
                    time_fmt = dt.strftime("%b %d, %I:%M %p")
                    if dt < now:
                        time_fmt += " (OVERDUE)"
                except Exception:
                    time_fmt = "unknown time"
                lines.append(f"  #{rid}: {text} — {time_fmt}")
            return f"Active reminders ({len(rows)}):\n" + "\n".join(lines)

        elif action == "done":
            if not rem_id:
                return "Please specify reminder_id to mark as done."
            store.mark_reminder_done(rem_id)
            return f"Reminder #{rem_id} marked as done."

        elif action == "set":
            if not message:
                return "Please specify a message for the reminder."
            trigger_at = _parse_future_time(time_str, minutes)
            rid = store.add_reminder(message, trigger_at)
            dt = datetime.fromtimestamp(trigger_at)
            time_fmt = dt.strftime("%b %d at %I:%M %p")

            # Also set a background thread for toast notification
            import threading
            delay = max(0, trigger_at - datetime.now().timestamp())
            def _notify():
                import time as _time
                _time.sleep(delay)
                try:
                    _toast_notification(f"JARVIS Reminder", message)
                except Exception:
                    logger.info(f"[Reminder] {message}")
            threading.Thread(target=_notify, daemon=True).start()

            return f"Reminder #{rid} set for {time_fmt}: {message}"

        else:
            return f"Unknown reminder action: {action}. Use: set, list, done."

    except Exception as e:
        return f"Reminder error: {e}"


def _toast_notification(title: str, message: str):
    """Show a Windows toast notification."""
    try:
        # Try Windows 10+ toast via PowerShell
        ps_cmd = (
            f'[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, '
            f'ContentType=WindowsRuntime] > $null; '
            f'$template = [Windows.UI.Notifications.ToastNotificationManager]::'
            f'GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02); '
            f'$text = $template.GetElementsByTagName("text"); '
            f'$text[0].AppendChild($template.CreateTextNode("{title}")) > $null; '
            f'$text[1].AppendChild($template.CreateTextNode("{message}")) > $null; '
            f'$notifier = [Windows.UI.Notifications.ToastNotificationManager]::'
            f'CreateToastNotifier("JARVIS"); '
            f'$notifier.Show([Windows.UI.Notifications.ToastNotification]::new($template))'
        )
        subprocess.run(["powershell", "-Command", ps_cmd],
                       capture_output=True, timeout=10)
    except Exception:
        # Fallback: tkinter messagebox
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            messagebox.showinfo(title, message)
            root.destroy()
        except Exception:
            pass


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


# ══════════════════════════════════════════════════════════════════════════════
#  NEW TOOLS
# ══════════════════════════════════════════════════════════════════════════════


def _exec_task_manager(args: dict) -> str:
    """Manage persistent tasks/to-do list."""
    from datetime import datetime
    action = args.get("action", "list").lower()
    desc = args.get("description", "")
    due_str = args.get("due", "")
    task_id = args.get("task_id", 0)

    try:
        store = _get_sqlite_store()

        if action == "list":
            tasks = store.get_pending_tasks()
            if not tasks:
                return "No pending tasks. All clear, Sir."
            lines = []
            for t in tasks:
                tid = t["id"]
                d = t["description"]
                due = t.get("due_at")
                due_fmt = ""
                if due:
                    try:
                        due_fmt = f" — due {datetime.fromtimestamp(due).strftime('%b %d, %I:%M %p')}"
                    except Exception:
                        pass
                lines.append(f"  #{tid}: {d}{due_fmt}")
            return f"Pending tasks ({len(tasks)}):\n" + "\n".join(lines)

        elif action == "add":
            if not desc:
                return "Please provide a task description."
            due_at = None
            if due_str:
                due_at = _parse_future_time(due_str)
            tid = store.add_task(desc, due_at)
            result = f"Task #{tid} added: {desc}"
            if due_at:
                result += f" (due {datetime.fromtimestamp(due_at).strftime('%b %d, %I:%M %p')})"
            return result

        elif action == "complete":
            if not task_id:
                return "Please specify task_id to complete."
            store._conn.execute(
                "UPDATE tasks SET status = 'completed' WHERE id = ?", (task_id,))
            store._conn.commit()
            return f"Task #{task_id} marked as completed."

        elif action == "delete":
            if not task_id:
                return "Please specify task_id to delete."
            store._conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
            store._conn.commit()
            return f"Task #{task_id} deleted."

        else:
            return f"Unknown task action: {action}. Use: add, list, complete, delete."

    except Exception as e:
        return f"Task manager error: {e}"


def _exec_clipboard(args: dict) -> str:
    """Read or write system clipboard."""
    action = args.get("action", "read").lower()
    text = args.get("text", "")

    try:
        if action == "read":
            result = subprocess.run(
                ["powershell", "-Command", "Get-Clipboard"],
                capture_output=True, text=True, timeout=5,
            )
            clip = result.stdout.strip()
            return f"Clipboard contents:\n{clip}" if clip else "Clipboard is empty."

        elif action == "write":
            if not text:
                return "No text specified to copy."
            # Use PowerShell Set-Clipboard
            subprocess.run(
                ["powershell", "-Command", f"Set-Clipboard -Value '{text}'"],
                capture_output=True, timeout=5,
            )
            return f"Copied to clipboard: {text[:100]}{'...' if len(text)>100 else ''}"

        return f"Unknown clipboard action: {action}. Use: read, write."
    except Exception as e:
        return f"Clipboard error: {e}"


def _exec_process_manager(args: dict) -> str:
    """List, kill, or check running processes."""
    try:
        import psutil
    except ImportError:
        return "psutil not available."

    action = args.get("action", "list").lower()
    name = args.get("name", "").lower()
    sort_by = args.get("sort_by", "memory").lower()

    try:
        if action == "list":
            procs = []
            for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_info"]):
                try:
                    info = p.info
                    mem_mb = (info["memory_info"].rss / (1024 ** 2)) if info["memory_info"] else 0
                    procs.append({
                        "pid": info["pid"],
                        "name": info["name"],
                        "cpu": info["cpu_percent"] or 0,
                        "mem": mem_mb,
                    })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            if sort_by == "cpu":
                procs.sort(key=lambda x: x["cpu"], reverse=True)
            elif sort_by == "name":
                procs.sort(key=lambda x: x["name"].lower())
            else:
                procs.sort(key=lambda x: x["mem"], reverse=True)

            lines = [f"{'PID':>7} {'CPU%':>5} {'MEM MB':>7}  NAME"]
            for p in procs[:20]:
                lines.append(f"{p['pid']:>7} {p['cpu']:>5.1f} {p['mem']:>7.1f}  {p['name']}")
            return f"Top {min(20, len(procs))} processes (sorted by {sort_by}):\n" + "\n".join(lines)

        elif action == "kill":
            if not name:
                return "Specify a process name to kill."
            killed = 0
            for p in psutil.process_iter(["name"]):
                try:
                    if name in p.info["name"].lower():
                        p.kill()
                        killed += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            return f"Killed {killed} process(es) matching '{name}'." if killed else f"No process matching '{name}' found."

        elif action == "check":
            if not name:
                return "Specify a process name to check."
            found = []
            for p in psutil.process_iter(["name", "pid"]):
                try:
                    if name in p.info["name"].lower():
                        found.append(f"{p.info['name']} (PID {p.info['pid']})")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            if found:
                return f"Running ({len(found)}):\n" + "\n".join(found[:10])
            return f"No process matching '{name}' is running."

        return f"Unknown action: {action}. Use: list, kill, check."
    except Exception as e:
        return f"Process manager error: {e}"


def _exec_daily_briefing(args: dict) -> str:
    """Generate a comprehensive daily briefing."""
    from datetime import datetime
    city = args.get("city", "Tel Aviv")

    sections = []

    # Time
    now = datetime.now()
    sections.append(f"Date: {now.strftime('%A, %B %d, %Y')}")
    sections.append(f"Time: {now.strftime('%I:%M %p')}")

    # Weather
    try:
        weather = _exec_weather({"city": city})
        sections.append(f"Weather: {weather}")
    except Exception:
        sections.append("Weather: unavailable")

    # System
    try:
        from tools.system_monitor import get_system_summary
        sections.append(get_system_summary())
    except Exception:
        pass

    # Tasks
    try:
        store = _get_sqlite_store()
        tasks = store.get_pending_tasks()
        if tasks:
            sections.append(f"Pending tasks ({len(tasks)}):")
            for t in tasks[:5]:
                sections.append(f"  - {t['description']}")
        else:
            sections.append("No pending tasks.")
    except Exception:
        pass

    # Reminders
    try:
        store = _get_sqlite_store()
        rows = store._conn.execute(
            "SELECT text, trigger_at FROM reminders WHERE done = 0 ORDER BY trigger_at LIMIT 5"
        ).fetchall()
        if rows:
            sections.append(f"Upcoming reminders ({len(rows)}):")
            for r in rows:
                try:
                    dt = datetime.fromtimestamp(r["trigger_at"]).strftime("%I:%M %p")
                    sections.append(f"  - {r['text']} at {dt}")
                except Exception:
                    sections.append(f"  - {r['text']}")
        else:
            sections.append("No active reminders.")
    except Exception:
        pass

    # News
    try:
        news = _exec_news({"topic": "", "max_results": 3})
        sections.append(f"Headlines:\n{news}")
    except Exception:
        pass

    return "\n".join(sections)


def _exec_notes(args: dict) -> str:
    """Persistent notes using SQLite facts table."""
    action = args.get("action", "list").lower()
    title = args.get("title", "")
    content = args.get("content", "")
    query = args.get("query", "")

    try:
        store = _get_sqlite_store()

        if action == "save":
            if not title:
                return "Please provide a note title."
            if not content:
                return "Please provide note content."
            store.save_fact(f"note:{title}", content, category="notes")
            return f"Note saved: '{title}'"

        elif action == "list":
            facts = store.get_facts(category="notes")
            if not facts:
                return "No notes saved yet."
            lines = []
            for f in facts:
                key = f["key"].replace("note:", "", 1)
                val = f["value"]
                if len(val) > 60:
                    val = val[:57] + "..."
                lines.append(f"  - {key}: {val}")
            return f"Notes ({len(facts)}):\n" + "\n".join(lines)

        elif action == "search":
            if not query:
                return "Please provide a search query."
            facts = store.get_facts(category="notes")
            matches = [f for f in facts
                       if query.lower() in f["key"].lower() or query.lower() in f["value"].lower()]
            if not matches:
                return f"No notes matching '{query}'."
            lines = []
            for f in matches:
                key = f["key"].replace("note:", "", 1)
                lines.append(f"  - {key}: {f['value']}")
            return f"Found {len(matches)} note(s):\n" + "\n".join(lines)

        elif action == "delete":
            if not title:
                return "Please specify note title to delete."
            store._conn.execute("DELETE FROM facts WHERE key = ?", (f"note:{title}",))
            store._conn.commit()
            return f"Note '{title}' deleted."

        return f"Unknown notes action: {action}. Use: save, list, search, delete."
    except Exception as e:
        return f"Notes error: {e}"


def _exec_translate(args: dict) -> str:
    """Translate text using MyMemory free API."""
    text = args.get("text", "")
    target = args.get("target_language", "en").strip().lower()
    source = args.get("source_language", "").strip().lower()

    if not text:
        return "No text to translate."

    # Map common names to codes
    lang_map = {
        "english": "en", "hebrew": "he", "spanish": "es", "french": "fr",
        "german": "de", "arabic": "ar", "russian": "ru", "chinese": "zh",
        "japanese": "ja", "korean": "ko", "portuguese": "pt", "italian": "it",
        "dutch": "nl", "turkish": "tr", "polish": "pl", "swedish": "sv",
        "hindi": "hi", "thai": "th",
    }
    target = lang_map.get(target, target)
    source = lang_map.get(source, source) if source else ""

    try:
        import httpx
        import urllib.parse

        lang_pair = f"{source or 'auto'}|{target}"
        url = f"https://api.mymemory.translated.net/get?q={urllib.parse.quote(text[:500])}&langpair={lang_pair}"
        resp = httpx.get(url, timeout=10).json()

        translated = resp.get("responseData", {}).get("translatedText", "")
        if translated:
            return f"Translation ({source or 'auto'} → {target}): {translated}"
        return "Translation failed — no result."
    except Exception as e:
        return f"Translation error: {e}"


def _exec_news(args: dict) -> str:
    """Fetch news headlines via DuckDuckGo news search."""
    topic = args.get("topic", "")
    max_results = args.get("max_results", 5)

    query = topic or "top news today"

    try:
        from ddgs import DDGS
        results = DDGS().news(query, max_results=max_results, region="wt-wt")
        if not results:
            return f"No news found for: {query}"
        lines = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            source = r.get("source", "")
            date = r.get("date", "")[:10]
            body = r.get("body", "")[:100]
            lines.append(f"{i}. [{source}] {title}")
            if body:
                lines.append(f"   {body}...")
        return "\n".join(lines)
    except ImportError:
        # Fallback to web search
        return _exec_web_search({"query": f"{query} news", "max_results": max_results})
    except Exception as e:
        return f"News error: {e}"


def _exec_timer(args: dict) -> str:
    """Set a countdown timer with notification."""
    import threading

    minutes = args.get("minutes", 0)
    seconds = args.get("seconds", 0)
    label = args.get("label", "Timer")

    total_seconds = (minutes * 60) + seconds
    if total_seconds <= 0:
        return "Please specify minutes or seconds for the timer."

    def _countdown():
        import time as _t
        _t.sleep(total_seconds)
        try:
            _toast_notification("JARVIS Timer", f"{label}: Time's up!")
        except Exception:
            logger.info(f"[Timer] {label}: Time's up!")

    threading.Thread(target=_countdown, daemon=True).start()

    if minutes > 0 and seconds > 0:
        return f"Timer set: {label} — {minutes}m {seconds}s"
    elif minutes > 0:
        return f"Timer set: {label} — {minutes} minute{'s' if minutes != 1 else ''}"
    else:
        return f"Timer set: {label} — {seconds} second{'s' if seconds != 1 else ''}"


def _exec_window_manager(args: dict) -> str:
    """Manage open windows on Windows."""
    import ctypes
    import ctypes.wintypes

    action = args.get("action", "list").lower()
    target = args.get("window_name", "").lower()

    try:
        if action == "minimize_all":
            subprocess.run(
                ["powershell", "-Command",
                 "(New-Object -ComObject Shell.Application).MinimizeAll()"],
                capture_output=True, timeout=5,
            )
            return "All windows minimized."

        elif action == "restore_all":
            subprocess.run(
                ["powershell", "-Command",
                 "(New-Object -ComObject Shell.Application).UndoMinimizeAll()"],
                capture_output=True, timeout=5,
            )
            return "All windows restored."

        elif action == "list":
            result = subprocess.run(
                ["powershell", "-Command",
                 "Get-Process | Where-Object {$_.MainWindowTitle -ne ''} | "
                 "Select-Object Id, MainWindowTitle | Format-Table -AutoSize | Out-String"],
                capture_output=True, text=True, timeout=10,
            )
            output = result.stdout.strip()
            return f"Open windows:\n{output}" if output else "No visible windows found."

        elif action == "switch":
            if not target:
                return "Specify window_name to switch to."
            # Find and activate window using PowerShell
            ps = (
                f"$proc = Get-Process | Where-Object {{$_.MainWindowTitle -like '*{target}*'}} | "
                f"Select-Object -First 1; "
                f"if ($proc) {{ "
                f"  $sig = '[DllImport(\"user32.dll\")] public static extern bool SetForegroundWindow(IntPtr hWnd);'; "
                f"  Add-Type -MemberDefinition $sig -Name NativeMethods -Namespace Win32; "
                f"  [Win32.NativeMethods]::SetForegroundWindow($proc.MainWindowHandle) | Out-Null; "
                f"  Write-Output \"Switched to: $($proc.MainWindowTitle)\""
                f"}} else {{ Write-Output 'Window not found.' }}"
            )
            result = subprocess.run(
                ["powershell", "-Command", ps],
                capture_output=True, text=True, timeout=10,
            )
            return result.stdout.strip() or "Window not found."

        elif action in ("snap_left", "snap_right"):
            if not target and action.startswith("snap"):
                # Snap current window
                import pyautogui
                if action == "snap_left":
                    pyautogui.hotkey("win", "left")
                else:
                    pyautogui.hotkey("win", "right")
                return f"Snapped window {action.replace('snap_', '')}."

            # First switch, then snap
            _exec_window_manager({"action": "switch", "window_name": target})
            import time
            time.sleep(0.3)
            import pyautogui
            if action == "snap_left":
                pyautogui.hotkey("win", "left")
            else:
                pyautogui.hotkey("win", "right")
            return f"Snapped '{target}' to the {action.replace('snap_', '')}."

        return f"Unknown action: {action}. Use: list, switch, snap_left, snap_right, minimize_all, restore_all."
    except Exception as e:
        return f"Window manager error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
#  AGENT MANAGEMENT TOOLS
# ══════════════════════════════════════════════════════════════════════════════


def _get_agent_manager():
    """Get the singleton AgentManager instance."""
    from agent.agent_manager import AgentManager
    return AgentManager()


def _exec_spawn_agent(args: dict) -> str:
    """Spawn a new background agent."""
    name = args.get("name", "").strip()
    task = args.get("task", "").strip()
    task_type = args.get("task_type", "general").strip().lower()

    if not name:
        return "Agent name is required."
    if not task:
        return "Task description is required."

    # Build params from the various optional fields
    params = {}
    if args.get("command"):
        params["command"] = args["command"]
    if args.get("file_path"):
        params["file_path"] = args["file_path"]
    if args.get("query"):
        params["query"] = args["query"]
    if args.get("tool_name"):
        params["tool_name"] = args["tool_name"]
    if args.get("tool_args"):
        try:
            params["tool_args"] = json.loads(args["tool_args"])
        except (json.JSONDecodeError, TypeError):
            params["tool_args"] = {}
    if args.get("interval_seconds"):
        params["interval_seconds"] = int(args["interval_seconds"])
    if args.get("visible"):
        params["visible"] = args["visible"]

    mgr = _get_agent_manager()
    return mgr.spawn_agent(name, task, task_type, params)


def _exec_agent_status(args: dict) -> str:
    """Get agent status."""
    name = args.get("name", "")
    mgr = _get_agent_manager()
    return mgr.get_agent_status(name)


def _exec_agent_result(args: dict) -> str:
    """Get agent result."""
    name = args.get("name", "")
    if not name:
        return "Agent name is required."
    mgr = _get_agent_manager()
    return mgr.get_agent_result(name)


def _exec_stop_agent(args: dict) -> str:
    """Stop a running agent."""
    name = args.get("name", "")
    if not name:
        return "Agent name is required."
    mgr = _get_agent_manager()
    return mgr.stop_agent(name)


def _exec_agent_message(args: dict) -> str:
    """Send message between agents."""
    sender = args.get("sender", "jarvis")
    recipient = args.get("recipient", "")
    message = args.get("message", "")
    if not recipient:
        return "Recipient agent name is required."
    if not message:
        return "Message content is required."
    mgr = _get_agent_manager()
    return mgr.send_message(sender, recipient, message)


def _exec_remove_agent(args: dict) -> str:
    """Remove an agent."""
    name = args.get("name", "")
    if not name:
        return "Agent name is required."
    mgr = _get_agent_manager()
    return mgr.remove_agent(name)
