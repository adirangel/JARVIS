"""JARVIS Tkinter GUI — Iron Man HUD aesthetic.

Features:
- API key entry screen (first run)
- Animated waveform / audio visualizer
- Conversation log with typewriter effect
- Status indicator (ONLINE / LISTENING / SPEAKING / PROCESSING)
- Text input for typed commands
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import threading
import time
import tkinter as tk
from tkinter import font as tkfont
from typing import Callable, Optional

from loguru import logger

# Unicode BiDi control characters
_RLM = "\u200F"   # Right-to-Left Mark — sets paragraph direction to RTL
_LRM = "\u200E"   # Left-to-Right Mark — isolates LTR tokens inside RTL text

# Regex for LTR tokens that need LRM isolation inside Hebrew text
_LTR_TOKEN_RE = re.compile(
    r"https?://\S+"
    r"|\b\d{1,3}(?:\.\d{1,3}){3}\b"
    r"|\bv?\d+(?:\.\d+)+\b"
    r"|\b[A-Za-z][A-Za-z0-9._-]*\b"
)

# ── Color palette (Iron Man HUD) ─────────────────────────────────────────────
BG_COLOR = "#0a0e17"
PANEL_COLOR = "#0d1521"
ACCENT_COLOR = "#00b4d8"        # Cyan accent
ACCENT_DIM = "#005f73"
TEXT_COLOR = "#e0f7fa"
USER_COLOR = "#4fc3f7"
JARVIS_COLOR = "#00e5ff"
STATUS_ONLINE = "#00e676"
STATUS_SPEAKING = "#ffab40"
STATUS_PROCESSING = "#ff6e40"
BORDER_COLOR = "#1a3a5c"

CONFIG_PATH = os.path.join("config", "api_keys.json")


class JarvisGUI:
    """Main JARVIS GUI window."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("J.A.R.V.I.S")
        self.root.configure(bg=BG_COLOR)
        self.root.geometry("900x650")
        self.root.minsize(700, 500)

        # Icon (optional)
        try:
            self.root.iconbitmap("config/jarvis.ico")
        except Exception:
            pass

        # Fonts
        self._setup_fonts()

        # State
        self._status = "OFFLINE"
        self._audio_level = 0.0
        self._waveform_data = [0.0] * 60
        self._api_key: Optional[str] = None
        self._on_api_key: Optional[Callable[[str], None]] = None
        self._on_text_input: Optional[Callable[[str], None]] = None
        self._conversation_lines: list[tuple[str, str]] = []  # (role, text)
        self._typewriter_queue: list[tuple[str, str]] = []
        self._typewriter_active = False
        # Streaming buffers — tokens accumulate, displayed at turn_complete
        self._jarvis_buffer: str = ""
        self._user_buffer: str = ""

        # Load saved API key
        self._api_key = self._load_api_key()

        if self._api_key:
            self._build_main_screen()
        else:
            self._build_api_key_screen()

    @staticmethod
    def _pick_font_family(root: tk.Tk) -> str:
        """Pick the best available font with Hebrew support."""
        candidates = ["Segoe UI", "Arial", "Tahoma", "David", "Noto Sans Hebrew"]
        available = set(tkfont.families(root))
        for f in candidates:
            if f in available:
                return f
        return "Arial"

    def _setup_fonts(self):
        heb_font = self._pick_font_family(self.root) or "Arial"
        self.font_title = tkfont.Font(family="Consolas", size=20, weight="bold")
        self.font_status = tkfont.Font(family="Consolas", size=11)
        self.font_log = tkfont.Font(family=heb_font, size=11)
        self.font_input = tkfont.Font(family=heb_font, size=11)
        self.font_entry = tkfont.Font(family="Consolas", size=12)
        self.font_button = tkfont.Font(family="Consolas", size=11, weight="bold")

    # ── API Key Screen ────────────────────────────────────────────────────────

    def _build_api_key_screen(self):
        self._clear_root()

        frame = tk.Frame(self.root, bg=BG_COLOR)
        frame.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(
            frame, text="J.A.R.V.I.S", font=self.font_title,
            fg=ACCENT_COLOR, bg=BG_COLOR
        ).pack(pady=(0, 5))

        tk.Label(
            frame, text="Just A Rather Very Intelligent System",
            font=self.font_status, fg=ACCENT_DIM, bg=BG_COLOR
        ).pack(pady=(0, 30))

        tk.Label(
            frame, text="Enter Gemini API Key:",
            font=self.font_status, fg=TEXT_COLOR, bg=BG_COLOR
        ).pack(pady=(0, 8))

        self._key_entry = tk.Entry(
            frame, font=self.font_entry, width=45, show="•",
            bg=PANEL_COLOR, fg=TEXT_COLOR, insertbackground=TEXT_COLOR,
            relief="flat", bd=2, highlightbackground=BORDER_COLOR,
            highlightcolor=ACCENT_COLOR, highlightthickness=1
        )
        self._key_entry.pack(pady=(0, 15), ipady=8)
        self._key_entry.bind("<Return>", lambda e: self._submit_key())
        self._key_entry.focus_set()

        self._key_btn = tk.Button(
            frame, text="INITIALIZE", font=self.font_button,
            fg=BG_COLOR, bg=ACCENT_COLOR, activebackground=ACCENT_DIM,
            activeforeground=TEXT_COLOR, relief="flat", padx=30, pady=8,
            command=self._submit_key, cursor="hand2"
        )
        self._key_btn.pack(pady=(0, 10))

        self._key_error = tk.Label(
            frame, text="", font=self.font_log,
            fg="#ff5252", bg=BG_COLOR
        )
        self._key_error.pack()

        tk.Label(
            frame, text="Get your key at: ai.google.dev",
            font=self.font_log, fg=ACCENT_DIM, bg=BG_COLOR, cursor="hand2"
        ).pack(pady=(20, 0))

    def _submit_key(self):
        key = self._key_entry.get().strip()
        if not key:
            self._key_error.config(text="Please enter an API key.")
            return
        if len(key) < 20:
            self._key_error.config(text="API key seems too short.")
            return

        self._api_key = key
        self._save_api_key(key)
        self._build_main_screen()

        if self._on_api_key:
            self._on_api_key(key)

    # ── Main Screen ───────────────────────────────────────────────────────────

    def _build_main_screen(self):
        self._clear_root()

        # Header
        header = tk.Frame(self.root, bg=BG_COLOR, height=60)
        header.pack(fill="x", padx=15, pady=(10, 0))
        header.pack_propagate(False)

        tk.Label(
            header, text="J.A.R.V.I.S", font=self.font_title,
            fg=ACCENT_COLOR, bg=BG_COLOR
        ).pack(side="left")

        # Status indicator
        self._status_frame = tk.Frame(header, bg=BG_COLOR)
        self._status_frame.pack(side="right")
        self._status_dot = tk.Canvas(
            self._status_frame, width=12, height=12, bg=BG_COLOR,
            highlightthickness=0
        )
        self._status_dot.pack(side="left", padx=(0, 6))
        self._status_dot.create_oval(2, 2, 10, 10, fill=STATUS_ONLINE, outline="")
        self._status_label = tk.Label(
            self._status_frame, text="ONLINE", font=self.font_status,
            fg=STATUS_ONLINE, bg=BG_COLOR
        )
        self._status_label.pack(side="left")

        # Waveform canvas
        self._wave_canvas = tk.Canvas(
            self.root, bg=PANEL_COLOR, height=80, highlightthickness=1,
            highlightbackground=BORDER_COLOR
        )
        self._wave_canvas.pack(fill="x", padx=15, pady=(10, 0))

        # Conversation log
        log_frame = tk.Frame(self.root, bg=PANEL_COLOR, highlightthickness=1,
                             highlightbackground=BORDER_COLOR)
        log_frame.pack(fill="both", expand=True, padx=15, pady=(10, 0))

        self._log_text = tk.Text(
            log_frame, bg=PANEL_COLOR, fg=TEXT_COLOR, font=self.font_log,
            wrap="word", relief="flat", padx=12, pady=10,
            insertbackground=PANEL_COLOR, selectbackground=ACCENT_DIM,
            state="disabled", cursor="arrow"
        )
        self._log_text.pack(fill="both", expand=True)

        # Configure text tags
        self._log_text.tag_configure("user", foreground=USER_COLOR)
        self._log_text.tag_configure("jarvis", foreground=JARVIS_COLOR)
        self._log_text.tag_configure("system", foreground=ACCENT_DIM)
        self._log_text.tag_configure("cursor", foreground=ACCENT_COLOR)
        # RTL tags for Hebrew text
        self._log_text.tag_configure("user_rtl", foreground=USER_COLOR, justify="right")
        self._log_text.tag_configure("jarvis_rtl", foreground=JARVIS_COLOR, justify="right")

        # Input bar
        input_frame = tk.Frame(self.root, bg=BG_COLOR)
        input_frame.pack(fill="x", padx=15, pady=(8, 12))

        self._text_input = tk.Entry(
            input_frame, font=self.font_input, bg=PANEL_COLOR, fg=TEXT_COLOR,
            insertbackground=TEXT_COLOR, relief="flat",
            highlightbackground=BORDER_COLOR, highlightcolor=ACCENT_COLOR,
            highlightthickness=1
        )
        self._text_input.pack(side="left", fill="x", expand=True, ipady=8, padx=(0, 8))
        self._text_input.bind("<Return>", self._handle_text_input)
        self._text_input.insert(0, "Type a message...")
        self._text_input.bind("<FocusIn>", self._clear_placeholder)
        self._text_input.bind("<FocusOut>", self._restore_placeholder)

        send_btn = tk.Button(
            input_frame, text="SEND", font=self.font_button,
            fg=BG_COLOR, bg=ACCENT_COLOR, activebackground=ACCENT_DIM,
            relief="flat", padx=15, pady=4, command=lambda: self._handle_text_input(None),
            cursor="hand2"
        )
        send_btn.pack(side="right")

        # Welcome message
        self._append_log("system", "JARVIS online. Listening for voice or text input.\n")

        # Start animation loop
        self._animate_waveform()

    # ── Input handling ────────────────────────────────────────────────────────

    def _clear_placeholder(self, event):
        if self._text_input.get() == "Type a message...":
            self._text_input.delete(0, "end")

    def _restore_placeholder(self, event):
        if not self._text_input.get().strip():
            self._text_input.delete(0, "end")
            self._text_input.insert(0, "Type a message...")

    def _handle_text_input(self, event):
        text = self._text_input.get().strip()
        if not text or text == "Type a message...":
            return
        self._text_input.delete(0, "end")
        self._append_log("user", text)
        if self._on_text_input:
            self._on_text_input(text)

    # ── Log / Conversation display ────────────────────────────────────────────

    @staticmethod
    def _has_hebrew(text: str) -> bool:
        """Check if text contains Hebrew characters."""
        return any('\u0590' <= ch <= '\u05FF' for ch in text)

    @staticmethod
    def _lrm_wrap(text: str) -> str:
        """Wrap LTR tokens (URLs, IPs, English words) with LRM marks
        so they stay readable inside RTL text."""
        return _LTR_TOKEN_RE.sub(lambda m: f"{_LRM}{m.group(0)}{_LRM}", text)

    def _append_log(self, role: str, text: str, typewriter: bool = False):
        """Add a complete message to the conversation log. Thread-safe."""
        if not hasattr(self, '_log_text'):
            return

        def _do():
            self._log_text.config(state="normal")
            prefix = ""
            if role == "user":
                prefix = "You: "
            elif role == "jarvis":
                prefix = "JARVIS: "
            elif role == "system":
                prefix = ">>> "

            is_rtl = self._has_hebrew(text)
            clean = text.rstrip("\n")

            if is_rtl:
                # RLM at start sets paragraph direction to RTL.
                # LRM wraps around the English prefix to isolate it.
                # LTR tokens inside Hebrew are also LRM-wrapped.
                msg = self._lrm_wrap(clean)
                if prefix:
                    line = f"{_RLM}{_LRM}{prefix}{_LRM}{msg}\n"
                else:
                    line = f"{_RLM}{msg}\n"
                tag = "user_rtl" if role == "user" else (
                    "jarvis_rtl" if role == "jarvis" else role
                )
            else:
                line = prefix + clean + "\n"
                tag = role

            self._log_text.insert("end", line, tag)
            self._log_text.see("end")
            self._log_text.config(state="disabled")

        self.root.after(0, _do)

    def append_token(self, token: str):
        """Buffer a streaming JARVIS token (displayed at turn_complete)."""
        self._jarvis_buffer += token

    def append_user_token(self, token: str):
        """Buffer a streaming user transcription token (displayed at turn_complete)."""
        self._user_buffer += token

    def finish_jarvis_line(self):
        """Flush JARVIS buffer to display as a complete line. Thread-safe."""
        text = self._jarvis_buffer.strip()
        self._jarvis_buffer = ""
        if text:
            self._append_log("jarvis", text)

    def finish_user_line(self):
        """Flush user buffer to display as a complete line. Thread-safe."""
        text = self._user_buffer.strip()
        self._user_buffer = ""
        if text:
            self._append_log("user", text)

    # ── Status ────────────────────────────────────────────────────────────────

    def set_status(self, status: str):
        """Update the status indicator. Thread-safe."""
        self._status = status
        colors = {
            "ONLINE": STATUS_ONLINE, "LISTENING": STATUS_ONLINE,
            "SPEAKING": STATUS_SPEAKING, "PROCESSING": STATUS_PROCESSING,
            "CONNECTING": STATUS_PROCESSING, "RECONNECTING": STATUS_PROCESSING,
            "AUTH_ERROR": "#ff5252",
            "OFFLINE": "#ff5252",
        }
        color = colors.get(status, ACCENT_DIM)

        def _do():
            if hasattr(self, '_status_label'):
                self._status_label.config(text=status, fg=color)
                self._status_dot.delete("all")
                self._status_dot.create_oval(2, 2, 10, 10, fill=color, outline="")

        self.root.after(0, _do)

    def set_audio_level(self, level: float):
        """Update audio level for waveform. Thread-safe."""
        self._audio_level = min(level * 5, 1.0)  # Amplify for visual

    # ── Waveform animation ────────────────────────────────────────────────────

    def _animate_waveform(self):
        """Draw animated waveform on canvas. 30fps."""
        if not hasattr(self, '_wave_canvas'):
            return

        canvas = self._wave_canvas
        w = canvas.winfo_width() or 880
        h = canvas.winfo_height() or 80
        canvas.delete("all")

        # Shift waveform data left
        self._waveform_data.pop(0)
        if self._status in ("SPEAKING", "PROCESSING"):
            val = self._audio_level + random.uniform(0.05, 0.3)
        elif self._status == "LISTENING":
            val = self._audio_level * 0.5 + random.uniform(0.01, 0.05)
        else:
            val = random.uniform(0.01, 0.03)
        self._waveform_data.append(min(val, 1.0))

        # Draw bars
        bar_count = len(self._waveform_data)
        bar_w = max(w / bar_count - 2, 2)
        mid_y = h / 2

        for i, v in enumerate(self._waveform_data):
            x = i * (w / bar_count)
            bar_h = max(v * mid_y * 0.9, 1)

            # Gradient color based on height
            intensity = min(int(v * 255), 255)
            color = f"#{0:02x}{min(180 + intensity // 3, 255):02x}{min(216 + intensity // 6, 255):02x}"

            canvas.create_rectangle(
                x, mid_y - bar_h, x + bar_w, mid_y + bar_h,
                fill=color, outline=""
            )

        # Center line
        canvas.create_line(0, mid_y, w, mid_y, fill=ACCENT_DIM, width=1, dash=(3, 3))

        # Schedule next frame (~30fps)
        self.root.after(33, self._animate_waveform)

    # ── API key persistence ───────────────────────────────────────────────────

    def _load_api_key(self) -> Optional[str]:
        try:
            if os.path.exists(CONFIG_PATH):
                with open(CONFIG_PATH, "r") as f:
                    data = json.load(f)
                    key = data.get("gemini_api_key", "")
                    if key and self._is_gemini_key(key):
                        return key
        except Exception:
            pass

        # Fallback: check api_key.txt (only if it looks like a Gemini key)
        try:
            if os.path.exists("api_key.txt"):
                with open("api_key.txt", "r") as f:
                    key = f.read().strip()
                    if key and self._is_gemini_key(key):
                        return key
        except Exception:
            pass

        return None

    @staticmethod
    def _is_gemini_key(key: str) -> bool:
        """Check if a key looks like a Gemini API key (starts with AIza)."""
        return key.startswith("AIza") and len(key) > 20

    def show_api_key_screen(self, error_msg: str = ""):
        """Switch to API key entry screen (e.g. after auth failure). Thread-safe."""
        def _do():
            self._api_key = None
            self._build_api_key_screen()
            if error_msg and hasattr(self, '_key_error'):
                self._key_error.config(text=error_msg)
        self.root.after(0, _do)

    def _save_api_key(self, key: str):
        os.makedirs("config", exist_ok=True)
        data = {}
        try:
            if os.path.exists(CONFIG_PATH):
                with open(CONFIG_PATH, "r") as f:
                    data = json.load(f)
        except Exception:
            pass
        data["gemini_api_key"] = key
        with open(CONFIG_PATH, "w") as f:
            json.dump(data, f, indent=2)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _clear_root(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    @property
    def api_key(self) -> Optional[str]:
        return self._api_key

    def on_api_key_ready(self, callback: Callable[[str], None]):
        self._on_api_key = callback

    def on_text_input(self, callback: Callable[[str], None]):
        self._on_text_input = callback

    def run(self):
        """Start the Tkinter main loop."""
        self.root.mainloop()
