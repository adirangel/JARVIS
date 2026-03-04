"""Browser automation for JARVIS — uses the user's real Chrome/Edge browser.

Instead of Playwright (separate browser instance), this opens the user's
actual default browser and uses pyautogui for keyboard/mouse interaction.
This lets JARVIS type in any chat, fill forms, and control real browser tabs.
"""

from __future__ import annotations

import os
import subprocess
import time
import webbrowser
from typing import Optional

import pyautogui
from loguru import logger

# Safety settings for pyautogui
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1

# ── Chrome / Edge path discovery (Windows) ────────────────────────────────────

_BROWSER_PATHS = [
    # Chrome
    os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
    os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
    os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe"),
    # Edge
    os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"),
    os.path.expandvars(r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"),
]


def _find_browser() -> Optional[str]:
    """Find Chrome or Edge executable on the system."""
    for path in _BROWSER_PATHS:
        if os.path.isfile(path):
            return path
    # Try 'where' command as fallback
    for cmd in ["chrome", "chrome.exe", "msedge", "msedge.exe"]:
        try:
            result = subprocess.run(
                ["where", cmd], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                path = result.stdout.strip().split("\n")[0]
                if os.path.isfile(path):
                    return path
        except Exception:
            continue
    return None


_browser_path: Optional[str] = None
_browser_launched = False


def _open_browser_url(url: str) -> str:
    """Open a URL in the user's real browser."""
    global _browser_path, _browser_launched

    if not url.startswith("http"):
        url = "https://" + url

    # Try to find Chrome/Edge path
    if _browser_path is None:
        _browser_path = _find_browser()

    if _browser_path:
        try:
            subprocess.Popen([_browser_path, url])
            _browser_launched = True
            return f"Opened {url} in browser"
        except Exception as e:
            logger.warning(f"Direct browser launch failed: {e}, falling back to webbrowser")

    # Fallback to Python's webbrowser module (uses system default)
    webbrowser.open(url)
    _browser_launched = True
    return f"Opened {url} in default browser"


def _wait_for_browser(seconds: float = 1.5):
    """Wait for browser to load/respond."""
    time.sleep(seconds)


def browser_action(
    action: str,
    url: str = "",
    query: str = "",
    selector: str = "",
    text: str = "",
    direction: str = "down",
) -> str:
    """Execute a browser action using the user's real browser + pyautogui.

    Actions:
        go_to      - Navigate to URL
        search     - Google search
        click      - Click at current mouse position or coordinates in selector "x,y"
        type       - Type text into the currently focused field
        type_and_enter - Type text and press Enter (for chat interfaces)
        scroll     - Scroll up/down
        new_tab    - Open a new tab
        close_tab  - Close current tab
        next_tab   - Switch to next tab
        prev_tab   - Switch to previous tab
        back       - Go back
        forward    - Go forward
        refresh    - Refresh page
        address    - Focus address bar and type URL
        find       - Open find dialog and search for text
        select_all - Select all text in current field
        copy       - Copy selection
        paste      - Paste clipboard
        close      - Close browser window
        focus      - Bring browser to front
    """
    try:
        if action == "go_to":
            if not url:
                return "No URL specified."
            return _open_browser_url(url)

        elif action == "search":
            if not query:
                return "No search query specified."
            search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            return _open_browser_url(search_url)

        elif action == "click":
            if selector and "," in selector:
                try:
                    parts = selector.split(",")
                    x, y = int(parts[0].strip()), int(parts[1].strip())
                    pyautogui.click(x, y)
                    return f"Clicked at ({x}, {y})"
                except (ValueError, IndexError):
                    pass
            # Click at current mouse position
            pyautogui.click()
            return "Clicked at current position"

        elif action == "type":
            if not text:
                return "No text to type."
            _type_text(text)
            return "Typed text into active field"

        elif action == "type_and_enter":
            if not text:
                return "No text to type."
            _type_text(text)
            time.sleep(0.1)
            pyautogui.press("enter")
            return "Typed text and pressed Enter"

        elif action == "scroll":
            amount = -5 if direction == "down" else 5
            pyautogui.scroll(amount)
            return f"Scrolled {direction}"

        elif action == "new_tab":
            pyautogui.hotkey("ctrl", "t")
            _wait_for_browser(0.5)
            if url:
                pyautogui.typewrite(url, interval=0.02)
                pyautogui.press("enter")
                return f"Opened new tab with {url}"
            return "Opened new tab"

        elif action == "close_tab":
            pyautogui.hotkey("ctrl", "w")
            return "Closed current tab"

        elif action == "next_tab":
            pyautogui.hotkey("ctrl", "tab")
            return "Switched to next tab"

        elif action == "prev_tab":
            pyautogui.hotkey("ctrl", "shift", "tab")
            return "Switched to previous tab"

        elif action == "back":
            pyautogui.hotkey("alt", "left")
            return "Went back"

        elif action == "forward":
            pyautogui.hotkey("alt", "right")
            return "Went forward"

        elif action == "refresh":
            pyautogui.press("f5")
            return "Refreshed page"

        elif action == "address":
            # Focus the address bar and type
            pyautogui.hotkey("ctrl", "l")
            _wait_for_browser(0.3)
            if url:
                pyautogui.typewrite(url, interval=0.02)
                pyautogui.press("enter")
                return f"Navigated to {url}"
            return "Focused address bar"

        elif action == "find":
            pyautogui.hotkey("ctrl", "f")
            _wait_for_browser(0.3)
            if text:
                _type_text(text)
                return f"Searching page for: {text}"
            return "Opened find dialog"

        elif action == "select_all":
            pyautogui.hotkey("ctrl", "a")
            return "Selected all"

        elif action == "copy":
            pyautogui.hotkey("ctrl", "c")
            return "Copied selection"

        elif action == "paste":
            pyautogui.hotkey("ctrl", "v")
            return "Pasted clipboard"

        elif action == "close":
            pyautogui.hotkey("alt", "F4")
            return "Closed browser window"

        elif action == "focus":
            _bring_browser_to_front()
            return "Browser brought to front"

        else:
            return f"Unknown browser action: {action}"

    except Exception as e:
        logger.error(f"Browser error: {e}")
        return f"Browser error: {e}"


def _type_text(text: str):
    """Type text supporting Unicode (Hebrew, etc.) via clipboard paste."""
    try:
        # Use clipboard method for Unicode text — works for Hebrew, etc.
        # Save current clipboard
        try:
            old_clip = subprocess.run(
                ["powershell", "-command", "Get-Clipboard"],
                capture_output=True, text=True, timeout=3
            ).stdout.rstrip("\n")
        except Exception:
            old_clip = ""

        # Set clipboard to our text (escape single quotes for PowerShell)
        escaped = text.replace("'", "''")
        subprocess.run(
            ["powershell", "-command", f"Set-Clipboard -Value '{escaped}'"],
            timeout=3
        )
        time.sleep(0.05)

        # Paste it
        pyautogui.hotkey("ctrl", "v")
        time.sleep(0.1)

        # Restore old clipboard (best effort)
        try:
            if old_clip:
                old_escaped = old_clip.replace("'", "''")
                subprocess.run(
                    ["powershell", "-command", f"Set-Clipboard -Value '{old_escaped}'"],
                    timeout=3
                )
        except Exception:
            pass

    except Exception:
        # Fallback: direct typing (ASCII only)
        pyautogui.typewrite(text, interval=0.03)


def _bring_browser_to_front():
    """Attempt to bring the browser window to the foreground."""
    try:
        ps_script = (
            '$chrome = Get-Process chrome -ErrorAction SilentlyContinue | Select-Object -First 1; '
            '$edge = Get-Process msedge -ErrorAction SilentlyContinue | Select-Object -First 1; '
            '$proc = if ($chrome) { $chrome } elseif ($edge) { $edge } else { $null }; '
            'if ($proc) { '
            'Add-Type -TypeDefinition @"\n'
            'using System; using System.Runtime.InteropServices;\n'
            'public class Win32Focus {\n'
            '  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);\n'
            '  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);\n'
            '}\n"@\n'
            '[Win32Focus]::ShowWindow($proc.MainWindowHandle, 9); '
            '[Win32Focus]::SetForegroundWindow($proc.MainWindowHandle) }'
        )
        subprocess.run(
            ["powershell", "-command", ps_script],
            timeout=5, capture_output=True
        )
    except Exception as e:
        logger.debug(f"Could not bring browser to front: {e}")


def close_browser():
    """No persistent browser to close (uses system browser)."""
    pass
