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
    key: str = "",
    keys: str = "",
    seconds: float = 0,
    question: str = "",
) -> str:
    """Execute a browser action using the user's real browser + pyautogui.

    Actions:
        go_to          - Navigate to URL (focuses browser, waits for load)
        search         - Google search
        click          - Click at current mouse position or coordinates in selector "x,y"
        double_click   - Double-click at position
        type           - Type text into the currently focused field
        type_and_enter - Type text and press Enter (for chat interfaces)
        press_key      - Press a single key (tab, enter, escape, space, up, down, etc.)
        hotkey         - Press key combination (ctrl+a, ctrl+enter, alt+tab, etc.)
        scroll         - Scroll up/down
        wait           - Wait for specified seconds (for page loads)
        screenshot     - Take a screenshot and describe what's visible on screen
        new_tab        - Open a new tab
        close_tab      - Close current tab
        next_tab       - Switch to next tab
        prev_tab       - Switch to previous tab
        back           - Go back
        forward        - Go forward
        refresh        - Refresh page
        address        - Focus address bar and type URL
        find           - Open find dialog and search for text
        select_all     - Select all text in current field
        copy           - Copy selection
        paste          - Paste clipboard
        close          - Close browser window
        focus          - Bring browser to front
    """
    try:
        if action == "go_to":
            if not url:
                return "No URL specified."
            _bring_browser_to_front()
            result = _open_browser_url(url)
            _wait_for_browser(2.0)  # Wait for page to load
            return result

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

        elif action == "double_click":
            if selector and "," in selector:
                try:
                    parts = selector.split(",")
                    x, y = int(parts[0].strip()), int(parts[1].strip())
                    pyautogui.doubleClick(x, y)
                    return f"Double-clicked at ({x}, {y})"
                except (ValueError, IndexError):
                    pass
            pyautogui.doubleClick()
            return "Double-clicked at current position"

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

        elif action == "press_key":
            target_key = key or text or ""
            if not target_key:
                return "No key specified. Use key parameter (e.g. 'tab', 'enter', 'escape')."
            target_key = target_key.strip().lower()
            pyautogui.press(target_key)
            return f"Pressed {target_key}"

        elif action == "hotkey":
            combo = keys or key or text or ""
            if not combo:
                return "No key combination specified. Use keys parameter (e.g. 'ctrl+a', 'ctrl+enter')."
            parts = [k.strip().lower() for k in combo.split("+")]
            pyautogui.hotkey(*parts)
            return f"Pressed {'+'.join(parts)}"

        elif action == "scroll":
            amount = -5 if direction == "down" else 5
            pyautogui.scroll(amount)
            return f"Scrolled {direction}"

        elif action == "wait":
            wait_time = seconds if seconds > 0 else 2.0
            wait_time = min(wait_time, 30.0)  # Cap at 30 seconds
            time.sleep(wait_time)
            return f"Waited {wait_time:.1f} seconds"

        elif action == "screenshot":
            return _take_and_describe_screenshot(question)

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


def _take_and_describe_screenshot(question: str = "") -> str:
    """Take a screenshot, send to Gemini Vision for description."""
    try:
        from PIL import ImageGrab
        import base64
        import io

        img = ImageGrab.grab()

        # Convert to bytes for Gemini
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        # Try Gemini Vision to describe the screenshot
        try:
            from google import genai
            from google.genai import types as gtypes

            # Load API key
            api_key = None
            for path in ["config/api_keys.json", "api_key.txt"]:
                try:
                    if path.endswith(".json"):
                        import json
                        with open(path) as f:
                            api_key = json.load(f).get("gemini_api_key", "")
                    else:
                        with open(path) as f:
                            api_key = f.read().strip()
                    if api_key:
                        break
                except Exception:
                    continue

            if not api_key:
                return "Screenshot taken but no API key for vision analysis."

            client = genai.Client(api_key=api_key)

            prompt = (
                "You are helping a voice assistant control a computer browser. "
                "Describe what you see on this screenshot concisely. Focus on: "
                "1) What website/app is open, 2) Key interactive elements visible "
                "(buttons, text fields, links), 3) Where the main input/text field is "
                "(describe its position: top/center/bottom, left/center/right). "
                "4) Any notable content or text on the page.\n"
            )
            if question:
                prompt += f"\nSpecifically answer: {question}\n"
            prompt += "\nBe concise — max 3-4 sentences."

            response = client.models.generate_content(
                model="gemini-2.5-flash-preview-05-20",
                contents=[
                    prompt,
                    gtypes.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                ],
            )
            description = response.text or "Could not describe the screenshot."
            return f"Screenshot analysis:\n{description}"

        except Exception as e:
            logger.warning(f"Vision analysis failed: {e}")
            # Fallback: save and report
            img.save("data/last_screenshot.png")
            return "Screenshot saved to data/last_screenshot.png (vision analysis unavailable)."

    except Exception as e:
        return f"Screenshot error: {e}"


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
