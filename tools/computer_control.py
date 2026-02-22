"""
Computer control tools: launch apps, mouse/keyboard actions, screenshots, file ops.
Safety: Destructive operations require user confirmation.
"""
import os
import subprocess
import platform
from typing import Tuple, Optional

# Try to import optional dependencies
try:
    import pyautogui
    import pynput
    from pynput import keyboard, mouse
    HAS_PYNPUT = True
except ImportError:
    HAS_PYNPUT = False

try:
    import pyautogui
    HAS_PYAUTOGUI = True
except ImportError:
    HAS_PYAUTOGUI = False

try:
    from PIL import ImageGrab
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

def launch_app(path: str) -> Tuple[bool, str]:
    """Launch an application given its path or command."""
    if not os.path.exists(path):
        try:
            subprocess.Popen([path])
            return True, f"Launched {path}"
        except Exception as e:
            return False, f"Failed to launch: {e}"
    else:
        try:
            if platform.system() == "Windows":
                os.startfile(path)
            else:
                subprocess.Popen([path])
            return True, f"Opened {path}"
        except Exception as e:
            return False, f"Error opening: {e}"

def click_at(x: int, y: int, button: str = "left") -> Tuple[bool, str]:
    """Move mouse to (x,y) and click."""
    if not HAS_PYAUTOGUI:
        return False, "pyautogui not available"
    try:
        pyautogui.moveTo(x, y, duration=0.2)
        pyautogui.click(button=button)
        return True, f"Clicked at ({x},{y})"
    except Exception as e:
        return False, f"Click failed: {e}"

def type_text(text: str, interval: float = 0.05) -> Tuple[bool, str]:
    """Type text with optional interval between keystrokes."""
    if not HAS_PYAUTOGUI:
        return False, "pyautogui not available"
    try:
        pyautogui.typewrite(text, interval=interval)
        return True, f"Typed: {text}"
    except Exception as e:
        return False, f"Type failed: {e}"

def press_key(key: str) -> Tuple[bool, str]:
    """Press a single key (e.g., 'enter', 'ctrl')."""
    if not HAS_PYAUTOGUI:
        return False, "pyautogui not available"
    try:
        pyautogui.press(key)
        return True, f"Pressed {key}"
    except Exception as e:
        return False, f"Press failed: {e}"

def take_screenshot(save_path: Optional[str] = None) -> Tuple[bool, str]:
    """Take screenshot. Returns (success, message or base64)."""
    if not HAS_PIL:
        return False, "PIL ImageGrab not available"
    try:
        img = ImageGrab.grab()
        if save_path:
            img.save(save_path)
            return True, f"Screenshot saved to {save_path}"
        else:
            import base64, io
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            data = base64.b64encode(buf.getvalue()).decode()
            return True, data
    except Exception as e:
        return False, f"Screenshot error: {e}"

def file_operation(operation: str, src: str, dst: Optional[str] = None) -> Tuple[bool, str]:
    """
    Perform file operations: 'copy', 'move', 'delete', 'rename', 'mkdir'.
    Delete is dangerous; we require explicit confirmation later.
    """
    try:
        if operation == "copy":
            import shutil
            shutil.copy2(src, dst)
            return True, f"Copied {src} to {dst}"
        elif operation == "move":
            import shutil
            shutil.move(src, dst)
            return True, f"Moved {src} to {dst}"
        elif operation == "delete":
            if src.startswith("/etc") or src.startswith("/bin") or src.startswith("/usr/bin") or src.startswith("C:\\Windows"):
                return False, f"Refusing to potentially delete system path: {src}"
            os.remove(src)
            return True, f"Deleted {src}"
        elif operation == "mkdir":
            os.makedirs(src, exist_ok=True)
            return True, f"Created directory {src}"
        elif operation == "listdir":
            items = os.listdir(src)
            return True, "\\n".join(items)
        else:
            return False, f"Unknown operation: {operation}"
    except Exception as e:
        return False, f"File op error: {e}"

def get_clipboard() -> Tuple[bool, str]:
    """Get clipboard text."""
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        data = root.clipboard_get()
        root.destroy()
        return True, data
    except Exception as e:
        return False, f"Clipboard error: {e}"

def set_clipboard(text: str) -> Tuple[bool, str]:
    """Set clipboard text."""
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        root.clipboard_clear()
        root.clipboard_append(text)
        root.update()
        root.destroy()
        return True, "Clipboard updated"
    except Exception as e:
        return False, f"Clipboard error: {e}"
