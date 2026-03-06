"""JARVIS Tkinter GUI — Iron Man HUD Mission Control.

Features:
- API key entry screen (first run)
- 3-column HUD: System Specs | Conversation + Waveform | Mission Control
- Live CPU / RAM / Disk / GPU / Network gauges
- Mission Control: JARVIS status, activity log, active tools
- Tasks & Reminders panel (reads from SQLite memory)
- Sub-agents status panel
- Animated waveform / audio visualizer
- Conversation log with Hebrew BiDi support
- Text input for typed commands
"""

from __future__ import annotations

import json
import math
import os
import platform
import random
import re
import subprocess
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import font as tkfont
from typing import Callable, Optional

from loguru import logger

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Unicode BiDi control characters
_RLM = "\u200F"
_LRM = "\u200E"
_LTR_TOKEN_RE = re.compile(
    r"https?://\S+"
    r"|\b\d{1,3}(?:\.\d{1,3}){3}\b"
    r"|\bv?\d+(?:\.\d+)+\b"
    r"|\b[A-Za-z][A-Za-z0-9._-]*\b"
)

# ── Color palette (Iron Man HUD) ─────────────────────────────────────────────
BG_COLOR = "#060a13"
BG_DARK = "#040810"
PANEL_COLOR = "#0b1120"
PANEL_HEADER = "#0d1528"
ACCENT_COLOR = "#00b4d8"
ACCENT_BRIGHT = "#00e5ff"
ACCENT_DIM = "#005f73"
ACCENT_GLOW = "#0088aa"
TEXT_COLOR = "#c8e6f0"
TEXT_DIM = "#5a7a8a"
TEXT_BRIGHT = "#e0f7fa"
USER_COLOR = "#4fc3f7"
JARVIS_COLOR = "#00e5ff"
STATUS_ONLINE = "#00e676"
STATUS_SPEAKING = "#ffab40"
STATUS_PROCESS = "#ff6e40"
BORDER_COLOR = "#0f2a44"
BORDER_GLOW = "#0d4466"
GAUGE_BG = "#0a1628"
GAUGE_GREEN = "#00e676"
GAUGE_YELLOW = "#ffab40"
GAUGE_RED = "#ff5252"
SKILL_BULB_ON = "#ffd54f"
SKILL_BULB_OFF = "#3e4451"
SKILL_BULB_GLOW = "#ffe082"
TASK_PENDING = "#ffab40"
TASK_DONE = "#00e676"
REMINDER_COLOR = "#ce93d8"
SUBAGENT_ACTIVE = "#00e676"
SUBAGENT_IDLE = "#37474f"
AGENT_RUNNING = "#4fc3f7"
AGENT_COMPLETED = "#00e676"
AGENT_FAILED = "#ff5252"
AGENT_STOPPED = "#78909c"
AGENT_PANEL_BG = "#081018"

CONFIG_PATH = os.path.join("config", "api_keys.json")


class JarvisGUI:
    """Main JARVIS GUI window — Mission Control HUD."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("J.A.R.V.I.S  \u2014  Mission Control")
        self.root.configure(bg=BG_COLOR)
        self.root.geometry("1400x800")
        self.root.minsize(1100, 650)

        try:
            self.root.iconbitmap("config/jarvis.ico")
        except Exception:
            pass

        self._setup_fonts()

        # ── State ─────────────────────────────────────────────────────────────
        self._status = "OFFLINE"
        self._audio_level = 0.0
        self._waveform_data = [0.0] * 60
        self._api_key: Optional[str] = None
        self._on_api_key: Optional[Callable[[str], None]] = None
        self._on_text_input: Optional[Callable[[str], None]] = None
        self._jarvis_buffer: str = ""
        self._user_buffer: str = ""
        self._start_time = time.time()

        # Activity log entries: list of (timestamp, message, category)
        self._activity_log: list[tuple[float, str, str]] = []
        # Sub-agents registry (system modules only)
        self._subagents: dict[str, dict] = {
            "Memory": {"status": "idle", "desc": "Long-term memory & recall"},
            "Tools": {"status": "idle", "desc": "Tool execution engine"},
            "Voice": {"status": "idle", "desc": "Speech I/O pipeline"},
            "Browser": {"status": "idle", "desc": "Web browser control"},
            "System": {"status": "idle", "desc": "System monitor & control"},
            "Skills": {"status": "idle", "desc": "Dynamic self-evolution"},
        }
        # User-spawned agents (separate from system modules)
        self._user_agents: dict[str, dict] = {}
        self._active_tools: list[str] = []
        self._active_skills: set[str] = set()  # currently executing dynamic skills
        self._loaded_skills_info: list[dict] = []  # [{name, desc}, ...]
        self._tasks: list[dict] = []
        self._reminders: list[dict] = []
        self._sys_stats: dict = {}
        self._pulse_phase = 0.0

        # Load saved API key
        self._api_key = self._load_api_key()

        if self._api_key:
            self._build_main_screen()
        else:
            self._build_api_key_screen()

    # ── Font setup ────────────────────────────────────────────────────────────

    @staticmethod
    def _pick_font_family(root: tk.Tk) -> str:
        candidates = ["Segoe UI", "Arial", "Tahoma", "David", "Noto Sans Hebrew"]
        available = set(tkfont.families(root))
        for f in candidates:
            if f in available:
                return f
        return "Arial"

    def _setup_fonts(self):
        heb = self._pick_font_family(self.root) or "Arial"
        self.font_title = tkfont.Font(family="Consolas", size=18, weight="bold")
        self.font_subtitle = tkfont.Font(family="Consolas", size=9)
        self.font_status = tkfont.Font(family="Consolas", size=10)
        self.font_log = tkfont.Font(family=heb, size=11)
        self.font_input = tkfont.Font(family=heb, size=11)
        self.font_entry = tkfont.Font(family="Consolas", size=12)
        self.font_button = tkfont.Font(family="Consolas", size=10, weight="bold")
        self.font_header = tkfont.Font(family="Consolas", size=10, weight="bold")
        self.font_small = tkfont.Font(family="Consolas", size=9)
        self.font_tiny = tkfont.Font(family="Consolas", size=8)
        self.font_gauge = tkfont.Font(family="Consolas", size=9, weight="bold")

    # ══════════════════════════════════════════════════════════════════════════
    #  API KEY SCREEN
    # ══════════════════════════════════════════════════════════════════════════

    def _build_api_key_screen(self):
        self._clear_root()

        frame = tk.Frame(self.root, bg=BG_COLOR)
        frame.place(relx=0.5, rely=0.5, anchor="center")

        # Arc reactor decoration
        reactor = tk.Canvas(frame, width=80, height=80, bg=BG_COLOR,
                            highlightthickness=0)
        reactor.pack(pady=(0, 15))
        self._draw_mini_reactor(reactor, 40, 40, 35)

        tk.Label(frame, text="J.A.R.V.I.S", font=self.font_title,
                 fg=ACCENT_BRIGHT, bg=BG_COLOR).pack(pady=(0, 3))
        tk.Label(frame, text="Just A Rather Very Intelligent System",
                 font=self.font_subtitle, fg=ACCENT_DIM, bg=BG_COLOR).pack(pady=(0, 5))
        tk.Label(frame, text="M I S S I O N   C O N T R O L",
                 font=self.font_tiny, fg=TEXT_DIM, bg=BG_COLOR).pack(pady=(0, 30))
        tk.Label(frame, text="ENTER GEMINI API KEY TO INITIALIZE:",
                 font=self.font_small, fg=TEXT_DIM, bg=BG_COLOR).pack(pady=(0, 8))

        self._key_entry = tk.Entry(
            frame, font=self.font_entry, width=45, show="\u2022",
            bg=PANEL_COLOR, fg=TEXT_COLOR, insertbackground=TEXT_COLOR,
            relief="flat", bd=2, highlightbackground=BORDER_COLOR,
            highlightcolor=ACCENT_COLOR, highlightthickness=1,
        )
        self._key_entry.pack(pady=(0, 15), ipady=8)
        self._key_entry.bind("<Return>", lambda _: self._submit_key())
        self._key_entry.focus_set()

        self._key_btn = tk.Button(
            frame, text="\u25b8  INITIALIZE  JARVIS", font=self.font_button,
            fg=BG_COLOR, bg=ACCENT_COLOR, activebackground=ACCENT_DIM,
            activeforeground=TEXT_COLOR, relief="flat", padx=30, pady=8,
            command=self._submit_key, cursor="hand2",
        )
        self._key_btn.pack(pady=(0, 10))

        self._key_error = tk.Label(frame, text="", font=self.font_small,
                                   fg="#ff5252", bg=BG_COLOR)
        self._key_error.pack()

        tk.Label(frame, text="Get your key at: ai.google.dev",
                 font=self.font_tiny, fg=ACCENT_DIM, bg=BG_COLOR,
                 cursor="hand2").pack(pady=(20, 0))

    @staticmethod
    def _draw_mini_reactor(canvas, cx, cy, r):
        canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                           outline=ACCENT_DIM, width=2)
        for ri, color in [(r * 0.7, ACCENT_GLOW), (r * 0.4, ACCENT_COLOR),
                          (r * 0.15, ACCENT_BRIGHT)]:
            canvas.create_oval(cx - ri, cy - ri, cx + ri, cy + ri,
                               outline=color, width=1)
        for angle in range(0, 360, 60):
            rad = math.radians(angle)
            x1 = cx + r * 0.4 * math.cos(rad)
            y1 = cy + r * 0.4 * math.sin(rad)
            x2 = cx + r * 0.7 * math.cos(rad)
            y2 = cy + r * 0.7 * math.sin(rad)
            canvas.create_line(x1, y1, x2, y2, fill=ACCENT_DIM, width=1)
        canvas.create_oval(cx - 3, cy - 3, cx + 3, cy + 3,
                           fill=ACCENT_BRIGHT, outline="")

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

    # ══════════════════════════════════════════════════════════════════════════
    #  MAIN SCREEN — 3-COLUMN HUD
    # ══════════════════════════════════════════════════════════════════════════

    def _build_main_screen(self):
        self._clear_root()
        self._start_time = time.time()

        # ── Top bar ───────────────────────────────────────────────────────
        topbar = tk.Frame(self.root, bg=BG_DARK, height=44)
        topbar.pack(fill="x")
        topbar.pack_propagate(False)

        tk.Label(topbar, text="  J.A.R.V.I.S", font=self.font_title,
                 fg=ACCENT_BRIGHT, bg=BG_DARK).pack(side="left", padx=(8, 0))
        tk.Label(topbar, text="MISSION CONTROL", font=self.font_tiny,
                 fg=TEXT_DIM, bg=BG_DARK).pack(side="left", padx=(12, 0), pady=(4, 0))

        # Right side: status
        self._status_frame = tk.Frame(topbar, bg=BG_DARK)
        self._status_frame.pack(side="right", padx=12)

        self._status_dot = tk.Canvas(self._status_frame, width=10, height=10,
                                     bg=BG_DARK, highlightthickness=0)
        self._status_dot.pack(side="left", padx=(0, 5))
        self._status_dot.create_oval(1, 1, 9, 9, fill=STATUS_ONLINE,
                                     outline="", tags="dot")

        self._status_label = tk.Label(self._status_frame, text="ONLINE",
                                      font=self.font_status, fg=STATUS_ONLINE, bg=BG_DARK)
        self._status_label.pack(side="left")

        self._uptime_label = tk.Label(topbar, text="UPTIME 00:00:00",
                                      font=self.font_tiny, fg=TEXT_DIM, bg=BG_DARK)
        self._uptime_label.pack(side="right", padx=(0, 20))

        # Separator
        tk.Frame(self.root, bg=BORDER_GLOW, height=1).pack(fill="x")

        # ── Body (3 columns) ─────────────────────────────────────────────
        body = tk.Frame(self.root, bg=BG_COLOR)
        body.pack(fill="both", expand=True)

        # LEFT — System Specs
        left_col = tk.Frame(body, bg=BG_COLOR, width=260)
        left_col.pack(side="left", fill="y", padx=(6, 0), pady=6)
        left_col.pack_propagate(False)
        self._build_left_panel(left_col)

        tk.Frame(body, bg=BORDER_COLOR, width=1).pack(side="left", fill="y", pady=8)

        # RIGHT — Mission Control
        right_col = tk.Frame(body, bg=BG_COLOR, width=280)
        right_col.pack(side="right", fill="y", padx=(0, 6), pady=6)
        right_col.pack_propagate(False)
        self._build_right_panel(right_col)

        tk.Frame(body, bg=BORDER_COLOR, width=1).pack(side="right", fill="y", pady=8)

        # CENTER — Conversation + Waveform
        center_col = tk.Frame(body, bg=BG_COLOR)
        center_col.pack(side="left", fill="both", expand=True, padx=4, pady=6)
        self._build_center_panel(center_col)

        # ── Bottom input bar ──────────────────────────────────────────────
        tk.Frame(self.root, bg=BORDER_GLOW, height=1).pack(fill="x")
        self._build_input_bar()

        # ── Start loops ──────────────────────────────────────────────────
        self._animate_waveform()
        self._update_system_stats()
        self._update_uptime()
        self._animate_pulse()
        self._update_tasks_reminders()
        self._poll_skills_periodic()

        self.log_activity("JARVIS Mission Control initialized", "system")
        self._append_log("system", "JARVIS online. All systems operational.\n")

    # ── LEFT PANEL: System Specs ──────────────────────────────────────────

    def _build_left_panel(self, parent):
        self._panel_header(parent, "\u2b21  SYSTEM DIAGNOSTICS")

        # Static system info
        info_frame = tk.Frame(parent, bg=PANEL_COLOR, highlightthickness=1,
                              highlightbackground=BORDER_COLOR)
        info_frame.pack(fill="x", pady=(0, 6))
        inner = tk.Frame(info_frame, bg=PANEL_COLOR)
        inner.pack(fill="x", padx=8, pady=6)

        os_name = f"{platform.system()} {platform.release()}"
        proc = platform.processor() or platform.machine()
        if len(proc) > 28:
            proc = proc[:25] + "..."

        for label, value in [("OS", os_name), ("ARCH", platform.machine()),
                             ("CPU", proc)]:
            row = tk.Frame(inner, bg=PANEL_COLOR)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=label, font=self.font_tiny, fg=TEXT_DIM,
                     bg=PANEL_COLOR, width=6, anchor="w").pack(side="left")
            tk.Label(row, text=value, font=self.font_tiny, fg=TEXT_COLOR,
                     bg=PANEL_COLOR, anchor="w").pack(side="left")

        # Gauges
        self._cpu_gauge = self._create_gauge(parent, "CPU USAGE")
        self._ram_gauge = self._create_gauge(parent, "MEMORY")
        self._disk_gauge = self._create_gauge(parent, "DISK")

        # Network
        self._net_frame = tk.Frame(parent, bg=PANEL_COLOR, highlightthickness=1,
                                   highlightbackground=BORDER_COLOR)
        self._net_frame.pack(fill="x", pady=(0, 6))
        net_inner = tk.Frame(self._net_frame, bg=PANEL_COLOR)
        net_inner.pack(fill="x", padx=8, pady=4)
        tk.Label(net_inner, text="NETWORK I/O", font=self.font_tiny,
                 fg=TEXT_DIM, bg=PANEL_COLOR).pack(anchor="w")
        self._net_label = tk.Label(net_inner, text="\u2191 0 MB  \u2193 0 MB",
                                   font=self.font_small, fg=ACCENT_COLOR,
                                   bg=PANEL_COLOR, anchor="w")
        self._net_label.pack(anchor="w")

        # GPU
        self._gpu_frame = tk.Frame(parent, bg=PANEL_COLOR, highlightthickness=1,
                                   highlightbackground=BORDER_COLOR)
        self._gpu_frame.pack(fill="x", pady=(0, 6))
        gpu_inner = tk.Frame(self._gpu_frame, bg=PANEL_COLOR)
        gpu_inner.pack(fill="x", padx=8, pady=4)
        tk.Label(gpu_inner, text="GPU", font=self.font_tiny, fg=TEXT_DIM,
                 bg=PANEL_COLOR).pack(anchor="w")
        self._gpu_label = tk.Label(gpu_inner, text="Detecting...",
                                   font=self.font_small, fg=ACCENT_COLOR,
                                   bg=PANEL_COLOR, anchor="w")
        self._gpu_label.pack(anchor="w")
        self._detect_gpu()

        # Sub-agents (system modules)
        self._panel_header(parent, "\u25c8  SYSTEM MODULES")
        self._agents_frame = tk.Frame(parent, bg=PANEL_COLOR, highlightthickness=1,
                                      highlightbackground=BORDER_COLOR)
        self._agents_frame.pack(fill="x", pady=(0, 6))
        self._refresh_subagents()

        # User-spawned agents
        self._panel_header(parent, "\u2726  LIVE AGENTS")
        self._user_agents_frame = tk.Frame(parent, bg=PANEL_COLOR, highlightthickness=1,
                                           highlightbackground=BORDER_COLOR)
        self._user_agents_frame.pack(fill="x", pady=(0, 6))

        # Live agent log (scrolling)
        self._agent_log_frame = tk.Frame(parent, bg=AGENT_PANEL_BG, highlightthickness=1,
                                         highlightbackground=BORDER_COLOR, height=120)
        self._agent_log_frame.pack(fill="x", pady=(0, 6))
        self._agent_log_frame.pack_propagate(False)

        self._agent_log_text = tk.Text(
            self._agent_log_frame, bg=AGENT_PANEL_BG, fg=TEXT_DIM,
            font=self.font_tiny, wrap="word", relief="flat",
            padx=6, pady=4, state="disabled", cursor="arrow",
            borderwidth=0, height=7,
        )
        self._agent_log_text.pack(fill="both", expand=True)
        self._agent_log_text.tag_configure("agent_name", foreground=AGENT_RUNNING, font=self.font_tiny)
        self._agent_log_text.tag_configure("agent_ok", foreground=AGENT_COMPLETED)
        self._agent_log_text.tag_configure("agent_err", foreground=AGENT_FAILED)
        self._agent_log_text.tag_configure("agent_info", foreground=TEXT_DIM)
        self._agent_log_text.tag_configure("agent_time", foreground=ACCENT_DIM)

        self._refresh_user_agents()
        self._update_agents_periodic()

    def _create_gauge(self, parent, label):
        frame = tk.Frame(parent, bg=PANEL_COLOR, highlightthickness=1,
                         highlightbackground=BORDER_COLOR)
        frame.pack(fill="x", pady=(0, 6))
        inner = tk.Frame(frame, bg=PANEL_COLOR)
        inner.pack(fill="x", padx=8, pady=4)

        header = tk.Frame(inner, bg=PANEL_COLOR)
        header.pack(fill="x")
        tk.Label(header, text=label, font=self.font_tiny, fg=TEXT_DIM,
                 bg=PANEL_COLOR).pack(side="left")
        val_label = tk.Label(header, text="0%", font=self.font_gauge,
                             fg=ACCENT_COLOR, bg=PANEL_COLOR)
        val_label.pack(side="right")

        bar_canvas = tk.Canvas(inner, height=8, bg=GAUGE_BG, highlightthickness=0)
        bar_canvas.pack(fill="x", pady=(3, 0))

        detail = tk.Label(inner, text="", font=self.font_tiny, fg=TEXT_DIM,
                          bg=PANEL_COLOR, anchor="w")
        detail.pack(anchor="w", pady=(2, 0))

        return {"canvas": bar_canvas, "value_label": val_label, "detail": detail}

    def _update_gauge(self, gauge, pct, detail_text=""):
        canvas = gauge["canvas"]
        if not canvas.winfo_exists():
            return
        canvas.delete("all")
        w = canvas.winfo_width() or 230
        h = canvas.winfo_height() or 8

        color = GAUGE_GREEN if pct < 60 else (GAUGE_YELLOW if pct < 85 else GAUGE_RED)
        canvas.create_rectangle(0, 0, w, h, fill=GAUGE_BG, outline="")
        fill_w = max(1, int(w * pct / 100))
        canvas.create_rectangle(0, 0, fill_w, h, fill=color, outline="")
        canvas.create_line(0, 0, fill_w, 0, fill=color, width=1)

        gauge["value_label"].config(text=f"{pct:.0f}%", fg=color)
        if detail_text:
            gauge["detail"].config(text=detail_text)

    # ── CENTER PANEL ──────────────────────────────────────────────────────

    def _build_center_panel(self, parent):
        # Conversation log
        log_frame = tk.Frame(parent, bg=PANEL_COLOR, highlightthickness=1,
                             highlightbackground=BORDER_COLOR)
        log_frame.pack(fill="both", expand=True, pady=(0, 4))

        log_header = tk.Frame(log_frame, bg=PANEL_HEADER, height=24)
        log_header.pack(fill="x")
        log_header.pack_propagate(False)
        tk.Label(log_header, text="  \u25b8 CONVERSATION LOG", font=self.font_tiny,
                 fg=TEXT_DIM, bg=PANEL_HEADER).pack(side="left")

        self._log_text = tk.Text(
            log_frame, bg=PANEL_COLOR, fg=TEXT_COLOR, font=self.font_log,
            wrap="word", relief="flat", padx=12, pady=8,
            insertbackground=PANEL_COLOR, selectbackground=ACCENT_DIM,
            state="disabled", cursor="arrow", borderwidth=0,
        )
        self._log_text.pack(fill="both", expand=True)

        scrollbar = tk.Scrollbar(self._log_text, orient="vertical",
                                 command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")

        self._log_text.tag_configure("user", foreground=USER_COLOR)
        self._log_text.tag_configure("jarvis", foreground=JARVIS_COLOR)
        self._log_text.tag_configure("system", foreground=ACCENT_DIM)
        self._log_text.tag_configure("user_rtl", foreground=USER_COLOR, justify="right")
        self._log_text.tag_configure("jarvis_rtl", foreground=JARVIS_COLOR, justify="right")
        self._log_text.tag_configure("tool_msg", foreground=STATUS_SPEAKING)

        # Waveform
        self._wave_canvas = tk.Canvas(parent, bg=PANEL_COLOR, height=55,
                                      highlightthickness=1,
                                      highlightbackground=BORDER_COLOR)
        self._wave_canvas.pack(fill="x")

    # ── RIGHT PANEL: Mission Control ──────────────────────────────────────

    def _build_right_panel(self, parent):
        # JARVIS core status
        self._panel_header(parent, "\u25c9  JARVIS CORE")
        status_frame = tk.Frame(parent, bg=PANEL_COLOR, highlightthickness=1,
                                highlightbackground=BORDER_COLOR)
        status_frame.pack(fill="x", pady=(0, 6))
        self._core_canvas = tk.Canvas(status_frame, width=250, height=60,
                                      bg=PANEL_COLOR, highlightthickness=0)
        self._core_canvas.pack(fill="x", padx=4, pady=4)

        # Activity log
        self._panel_header(parent, "\u25b8  ACTIVITY LOG")
        act_frame = tk.Frame(parent, bg=PANEL_COLOR, highlightthickness=1,
                             highlightbackground=BORDER_COLOR, height=140)
        act_frame.pack(fill="x", pady=(0, 6))
        act_frame.pack_propagate(False)

        self._activity_text = tk.Text(
            act_frame, bg=PANEL_COLOR, fg=TEXT_DIM, font=self.font_tiny,
            wrap="word", relief="flat", padx=6, pady=4,
            state="disabled", cursor="arrow", borderwidth=0, height=8,
        )
        self._activity_text.pack(fill="both", expand=True)
        self._activity_text.tag_configure("time", foreground=ACCENT_DIM)
        self._activity_text.tag_configure("system", foreground=ACCENT_COLOR)
        self._activity_text.tag_configure("tool", foreground=STATUS_SPEAKING)
        self._activity_text.tag_configure("error", foreground=GAUGE_RED)
        self._activity_text.tag_configure("info", foreground=TEXT_DIM)
        self._activity_text.tag_configure("agent", foreground=AGENT_RUNNING)

        # Tasks
        self._panel_header(parent, "\u2610  TASKS")
        self._tasks_frame = tk.Frame(parent, bg=PANEL_COLOR, highlightthickness=1,
                                     highlightbackground=BORDER_COLOR)
        self._tasks_frame.pack(fill="x", pady=(0, 6))
        self._tasks_inner = tk.Frame(self._tasks_frame, bg=PANEL_COLOR)
        self._tasks_inner.pack(fill="x", padx=6, pady=4)
        tk.Label(self._tasks_inner, text="No pending tasks", font=self.font_tiny,
                 fg=TEXT_DIM, bg=PANEL_COLOR).pack(anchor="w")

        # Reminders
        self._panel_header(parent, "\u23f1  REMINDERS")
        self._reminders_frame = tk.Frame(parent, bg=PANEL_COLOR, highlightthickness=1,
                                         highlightbackground=BORDER_COLOR)
        self._reminders_frame.pack(fill="x", pady=(0, 6))
        self._reminders_inner = tk.Frame(self._reminders_frame, bg=PANEL_COLOR)
        self._reminders_inner.pack(fill="x", padx=6, pady=4)
        tk.Label(self._reminders_inner, text="No active reminders",
                 font=self.font_tiny, fg=TEXT_DIM, bg=PANEL_COLOR).pack(anchor="w")

        # Active tools
        self._panel_header(parent, "\u26a1  ACTIVE TOOLS")
        self._tools_frame = tk.Frame(parent, bg=PANEL_COLOR, highlightthickness=1,
                                     highlightbackground=BORDER_COLOR)
        self._tools_frame.pack(fill="x", pady=(0, 6))
        self._tools_inner = tk.Frame(self._tools_frame, bg=PANEL_COLOR)
        self._tools_inner.pack(fill="x", padx=6, pady=4)
        tk.Label(self._tools_inner, text="No tools active", font=self.font_tiny,
                 fg=TEXT_DIM, bg=PANEL_COLOR).pack(anchor="w")

        # Learned Skills
        self._panel_header(parent, "\U0001f4a1  LEARNED SKILLS")
        self._skills_frame = tk.Frame(parent, bg=PANEL_COLOR, highlightthickness=1,
                                      highlightbackground=BORDER_COLOR)
        self._skills_frame.pack(fill="x", pady=(0, 6))
        self._skills_inner = tk.Frame(self._skills_frame, bg=PANEL_COLOR)
        self._skills_inner.pack(fill="x", padx=6, pady=4)
        tk.Label(self._skills_inner, text="No skills learned yet",
                 font=self.font_tiny, fg=TEXT_DIM, bg=PANEL_COLOR).pack(anchor="w")
        self._load_skills_list()

    # ── INPUT BAR ─────────────────────────────────────────────────────────

    def _build_input_bar(self):
        bar = tk.Frame(self.root, bg=BG_DARK, height=48)
        bar.pack(fill="x")
        bar.pack_propagate(False)

        inner = tk.Frame(bar, bg=BG_DARK)
        inner.pack(fill="x", padx=12, pady=8)

        tk.Label(inner, text="\u25b8", font=self.font_status, fg=ACCENT_COLOR,
                 bg=BG_DARK).pack(side="left", padx=(0, 5))

        self._text_input = tk.Entry(
            inner, font=self.font_input, bg=PANEL_COLOR, fg=TEXT_COLOR,
            insertbackground=TEXT_COLOR, relief="flat",
            highlightbackground=BORDER_COLOR, highlightcolor=ACCENT_COLOR,
            highlightthickness=1,
        )
        self._text_input.pack(side="left", fill="x", expand=True, ipady=6, padx=(0, 8))
        self._text_input.bind("<Return>", self._handle_text_input)
        self._text_input.insert(0, "Type a command...")
        self._text_input.config(fg=TEXT_DIM)
        self._text_input.bind("<FocusIn>", self._clear_placeholder)
        self._text_input.bind("<FocusOut>", self._restore_placeholder)

        tk.Button(
            inner, text="SEND", font=self.font_button,
            fg=BG_COLOR, bg=ACCENT_COLOR, activebackground=ACCENT_DIM,
            relief="flat", padx=15, pady=2,
            command=lambda: self._handle_text_input(None), cursor="hand2",
        ).pack(side="right")

    # ── panel header helper ───────────────────────────────────────────────

    def _panel_header(self, parent, text):
        hdr = tk.Frame(parent, bg=BG_COLOR)
        hdr.pack(fill="x", pady=(6, 2))
        tk.Label(hdr, text=text, font=self.font_header, fg=ACCENT_COLOR,
                 bg=BG_COLOR, anchor="w").pack(side="left")
        tk.Frame(hdr, bg=BORDER_COLOR, height=1).pack(
            side="left", fill="x", expand=True, padx=(8, 0), pady=6)

    # ══════════════════════════════════════════════════════════════════════════
    #  SYSTEM STATS
    # ══════════════════════════════════════════════════════════════════════════

    def _update_system_stats(self):
        if not PSUTIL_AVAILABLE:
            return

        def _fetch():
            try:
                cpu = psutil.cpu_percent(interval=0)
                mem = psutil.virtual_memory()
                disk_path = "C:\\" if platform.system() == "Windows" else "/"
                disk = psutil.disk_usage(disk_path)
                net = psutil.net_io_counters()
                freq = psutil.cpu_freq()
                return {
                    "cpu": cpu,
                    "cores": psutil.cpu_count(),
                    "freq": freq.current if freq else 0,
                    "mem_pct": mem.percent,
                    "mem_used": mem.used / (1024 ** 3),
                    "mem_total": mem.total / (1024 ** 3),
                    "disk_pct": disk.percent,
                    "disk_used": disk.used / (1024 ** 3),
                    "disk_total": disk.total / (1024 ** 3),
                    "net_sent": net.bytes_sent / (1024 ** 2),
                    "net_recv": net.bytes_recv / (1024 ** 2),
                }
            except Exception:
                return None

        def _apply(stats):
            if not stats or not hasattr(self, "_cpu_gauge"):
                return
            try:
                if not self._cpu_gauge["canvas"].winfo_exists():
                    return
            except Exception:
                return
            self._sys_stats = stats
            freq_str = f"{stats['freq']:.0f} MHz" if stats["freq"] else ""
            self._update_gauge(
                self._cpu_gauge, stats["cpu"],
                f"{stats['cores']} cores  \u2022  {freq_str}" if freq_str else f"{stats['cores']} cores",
            )
            self._update_gauge(
                self._ram_gauge, stats["mem_pct"],
                f"{stats['mem_used']:.1f} / {stats['mem_total']:.1f} GB",
            )
            self._update_gauge(
                self._disk_gauge, stats["disk_pct"],
                f"{stats['disk_used']:.0f} / {stats['disk_total']:.0f} GB",
            )
            self._net_label.config(
                text=f"\u2191 {stats['net_sent']:.0f} MB    \u2193 {stats['net_recv']:.0f} MB"
            ) if self._net_label.winfo_exists() else None

        def _work():
            stats = _fetch()
            if stats:
                try:
                    self.root.after(0, lambda: _apply(stats))
                except RuntimeError:
                    pass  # mainloop not ready yet

        threading.Thread(target=_work, daemon=True).start()
        self.root.after(2000, self._update_system_stats)

    def _detect_gpu(self):
        def _work():
            gpu_info = "Not detected"
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,memory.total",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0 and result.stdout.strip():
                    parts = result.stdout.strip().split(",")
                    name = parts[0].strip()
                    mem = parts[1].strip() if len(parts) > 1 else "?"
                    gpu_info = f"{name}\n{mem} MB VRAM"
            except Exception:
                pass
            if gpu_info == "Not detected" and platform.system() == "Windows":
                try:
                    result = subprocess.run(
                        ["wmic", "path", "win32_VideoController", "get", "name"],
                        capture_output=True, text=True, timeout=5,
                    )
                    lines = [l.strip() for l in result.stdout.strip().split("\n")
                             if l.strip() and l.strip() != "Name"]
                    if lines:
                        gpu_info = lines[0]
                except Exception:
                    pass

            def _apply():
                if hasattr(self, "_gpu_label") and self._gpu_label.winfo_exists():
                    self._gpu_label.config(text=gpu_info)

            try:
                self.root.after(0, _apply)
            except RuntimeError:
                pass  # mainloop not ready yet

        threading.Thread(target=_work, daemon=True).start()

    def _update_uptime(self):
        if not hasattr(self, "_uptime_label") or not self._uptime_label.winfo_exists():
            return
        elapsed = int(time.time() - self._start_time)
        h, m, s = elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60
        self._uptime_label.config(text=f"UPTIME {h:02d}:{m:02d}:{s:02d}")
        self.root.after(1000, self._update_uptime)

    # ══════════════════════════════════════════════════════════════════════════
    #  PULSE ANIMATION (JARVIS Core)
    # ══════════════════════════════════════════════════════════════════════════

    def _animate_pulse(self):
        if not hasattr(self, "_core_canvas") or not self._core_canvas.winfo_exists():
            return
        canvas = self._core_canvas
        canvas.delete("all")
        w = canvas.winfo_width() or 250
        h = canvas.winfo_height() or 60
        cx, cy = 30, h // 2

        self._pulse_phase += 0.08
        pulse = (math.sin(self._pulse_phase) + 1) / 2

        status_colors = {
            "ONLINE": STATUS_ONLINE, "LISTENING": STATUS_ONLINE,
            "SPEAKING": STATUS_SPEAKING, "PROCESSING": STATUS_PROCESS,
            "CONNECTING": STATUS_PROCESS, "RECONNECTING": STATUS_PROCESS,
        }
        color = status_colors.get(self._status, ACCENT_DIM)

        pr = 12 + pulse * 4
        canvas.create_oval(cx - pr, cy - pr, cx + pr, cy + pr,
                           outline=color, width=1)
        canvas.create_oval(cx - 6, cy - 6, cx + 6, cy + 6, fill=color, outline="")
        canvas.create_oval(cx - 3, cy - 3, cx + 3, cy + 3,
                           fill=ACCENT_BRIGHT, outline="")

        canvas.create_text(65, cy - 10, text=self._status, font=self.font_gauge,
                           fill=color, anchor="w")

        substatus_map = {
            "LISTENING": "Awaiting voice or text input",
            "SPEAKING": "Generating audio response",
            "PROCESSING": "Executing tool or thinking",
            "CONNECTING": "Establishing Gemini connection",
            "RECONNECTING": "Connection lost \u2014 reconnecting",
            "ONLINE": "All systems operational",
        }
        substatus = substatus_map.get(self._status, "Standing by")
        canvas.create_text(65, cy + 8, text=substatus, font=self.font_tiny,
                           fill=TEXT_DIM, anchor="w")

        self.root.after(50, self._animate_pulse)

    # ══════════════════════════════════════════════════════════════════════════
    #  ACTIVITY LOG
    # ══════════════════════════════════════════════════════════════════════════

    def log_activity(self, message: str, category: str = "info"):
        """Add an entry to the activity log. Thread-safe."""
        self._activity_log.append((time.time(), message, category))
        if len(self._activity_log) > 100:
            self._activity_log = self._activity_log[-100:]

        def _do():
            if not hasattr(self, "_activity_text"):
                return
            self._activity_text.config(state="normal")
            ts = datetime.now().strftime("%H:%M:%S")
            tag = category if category in ("system", "tool", "error", "info", "agent") else "info"
            self._activity_text.insert("end", f"[{ts}] ", "time")
            self._activity_text.insert("end", f"{message}\n", tag)
            self._activity_text.see("end")
            self._activity_text.config(state="disabled")

        self.root.after(0, _do)

    # ══════════════════════════════════════════════════════════════════════════
    #  TASKS & REMINDERS
    # ══════════════════════════════════════════════════════════════════════════

    def _update_tasks_reminders(self):
        def _load():
            tasks: list[dict] = []
            reminders: list[dict] = []
            try:
                from memory.sqlite_store import SQLiteStore
                store = SQLiteStore("data/jarvis.db")
                tasks = store.get_pending_tasks()
                rows = store._conn.execute(
                    "SELECT id, text, trigger_at FROM reminders WHERE done = 0 ORDER BY trigger_at"
                ).fetchall()
                reminders = [dict(r) for r in rows]
                store.close()
            except Exception:
                pass
            return tasks, reminders

        def _apply(tasks, reminders):
            if not hasattr(self, "_tasks_inner") or not self._tasks_inner.winfo_exists():
                return
            self._tasks = tasks
            self._reminders = reminders
            self._refresh_tasks_display()
            self._refresh_reminders_display()

        def _work():
            tasks, reminders = _load()
            try:
                self.root.after(0, lambda: _apply(tasks, reminders))
            except RuntimeError:
                pass  # mainloop not ready yet

        threading.Thread(target=_work, daemon=True).start()
        self.root.after(10000, self._update_tasks_reminders)

    def _refresh_tasks_display(self):
        if not hasattr(self, "_tasks_inner") or not self._tasks_inner.winfo_exists():
            return
        for w in self._tasks_inner.winfo_children():
            w.destroy()
        if not self._tasks:
            tk.Label(self._tasks_inner, text="No pending tasks", font=self.font_tiny,
                     fg=TEXT_DIM, bg=PANEL_COLOR).pack(anchor="w")
            return
        for task in self._tasks[:8]:
            row = tk.Frame(self._tasks_inner, bg=PANEL_COLOR)
            row.pack(fill="x", pady=1)
            tk.Label(row, text="\u2610", font=self.font_tiny, fg=TASK_PENDING,
                     bg=PANEL_COLOR).pack(side="left", padx=(0, 4))
            desc = task.get("description", "?")
            if len(desc) > 32:
                desc = desc[:29] + "..."
            tk.Label(row, text=desc, font=self.font_tiny, fg=TEXT_COLOR,
                     bg=PANEL_COLOR, anchor="w").pack(side="left")
            if task.get("due_at"):
                try:
                    due = datetime.fromtimestamp(task["due_at"]).strftime("%m/%d %H:%M")
                    tk.Label(row, text=due, font=self.font_tiny, fg=TEXT_DIM,
                             bg=PANEL_COLOR).pack(side="right")
                except Exception:
                    pass
        if len(self._tasks) > 8:
            tk.Label(self._tasks_inner,
                     text=f"  +{len(self._tasks) - 8} more...",
                     font=self.font_tiny, fg=TEXT_DIM, bg=PANEL_COLOR).pack(anchor="w")

    def _refresh_reminders_display(self):
        if not hasattr(self, "_reminders_inner") or not self._reminders_inner.winfo_exists():
            return
        for w in self._reminders_inner.winfo_children():
            w.destroy()
        if not self._reminders:
            tk.Label(self._reminders_inner, text="No active reminders",
                     font=self.font_tiny, fg=TEXT_DIM, bg=PANEL_COLOR).pack(anchor="w")
            return
        now = time.time()
        for rem in self._reminders[:6]:
            row = tk.Frame(self._reminders_inner, bg=PANEL_COLOR)
            row.pack(fill="x", pady=1)
            trigger = rem.get("trigger_at", 0)
            overdue = trigger and trigger <= now
            ic = GAUGE_RED if overdue else REMINDER_COLOR
            tk.Label(row, text="\u23f0" if overdue else "\u23f1", font=self.font_tiny,
                     fg=ic, bg=PANEL_COLOR).pack(side="left", padx=(0, 4))
            text = rem.get("text", "?")
            if len(text) > 30:
                text = text[:27] + "..."
            tk.Label(row, text=text, font=self.font_tiny,
                     fg=TEXT_COLOR if not overdue else GAUGE_RED,
                     bg=PANEL_COLOR, anchor="w").pack(side="left")
            if trigger:
                try:
                    tk.Label(row, text=datetime.fromtimestamp(trigger).strftime("%H:%M"),
                             font=self.font_tiny, fg=TEXT_DIM,
                             bg=PANEL_COLOR).pack(side="right")
                except Exception:
                    pass

    # ══════════════════════════════════════════════════════════════════════════
    #  SYSTEM MODULES (formerly SUB-AGENTS)
    # ══════════════════════════════════════════════════════════════════════════

    def _refresh_subagents(self):
        for w in self._agents_frame.winfo_children():
            w.destroy()
        inner = tk.Frame(self._agents_frame, bg=PANEL_COLOR)
        inner.pack(fill="x", padx=6, pady=4)
        for name, info in self._subagents.items():
            row = tk.Frame(inner, bg=PANEL_COLOR)
            row.pack(fill="x", pady=1)
            active = info["status"] == "active"
            dc = SUBAGENT_ACTIVE if active else SUBAGENT_IDLE
            tk.Label(row, text="\u25c9" if active else "\u25cb",
                     font=self.font_tiny, fg=dc, bg=PANEL_COLOR).pack(
                side="left", padx=(0, 4))
            tk.Label(row, text=name, font=self.font_tiny,
                     fg=TEXT_COLOR if active else TEXT_DIM,
                     bg=PANEL_COLOR).pack(side="left")
            tk.Label(row, text=info["desc"], font=self.font_tiny,
                     fg=TEXT_DIM, bg=PANEL_COLOR).pack(side="right")

    def set_subagent_status(self, name: str, status: str, desc: str = ""):
        """Update a system module's status. Thread-safe."""
        # Route user-spawned agents to separate panel
        if name.startswith("Agent:"):
            self.set_user_agent_status(name[6:], status, desc)
            return
        if name in self._subagents:
            self._subagents[name]["status"] = status
            if desc:
                self._subagents[name]["desc"] = desc
        else:
            self._subagents[name] = {"status": status, "desc": desc or name}

        def _do():
            if hasattr(self, "_agents_frame"):
                self._refresh_subagents()

        self.root.after(0, _do)

    # ══════════════════════════════════════════════════════════════════════════
    #  USER-SPAWNED AGENTS (live view)
    # ══════════════════════════════════════════════════════════════════════════

    def set_user_agent_status(self, name: str, status: str, desc: str = ""):
        """Update a user-spawned agent's status. Thread-safe."""
        self._user_agents[name] = {
            "status": status,
            "desc": desc or name,
            "updated": time.time(),
        }
        def _do():
            if hasattr(self, "_user_agents_frame"):
                self._refresh_user_agents()
        self.root.after(0, _do)

    def _refresh_user_agents(self):
        """Redraw the user-spawned agents list."""
        if not hasattr(self, "_user_agents_frame"):
            return
        for w in self._user_agents_frame.winfo_children():
            w.destroy()
        inner = tk.Frame(self._user_agents_frame, bg=PANEL_COLOR)
        inner.pack(fill="x", padx=6, pady=4)

        if not self._user_agents:
            tk.Label(inner, text="No agents spawned", font=self.font_tiny,
                     fg=TEXT_DIM, bg=PANEL_COLOR).pack(anchor="w")
            return

        for name, info in self._user_agents.items():
            row = tk.Frame(inner, bg=PANEL_COLOR)
            row.pack(fill="x", pady=1)
            status = info.get("status", "idle")

            # Status icon + color
            icon_map = {
                "running": ("\u25b6", AGENT_RUNNING),
                "active": ("\u25b6", AGENT_RUNNING),
                "completed": ("\u2714", AGENT_COMPLETED),
                "failed": ("\u2718", AGENT_FAILED),
                "stopped": ("\u25a0", AGENT_STOPPED),
                "idle": ("\u25cb", SUBAGENT_IDLE),
                "waiting": ("\u25cc", GAUGE_YELLOW),
            }
            icon, color = icon_map.get(status, ("\u25cb", SUBAGENT_IDLE))

            tk.Label(row, text=icon, font=self.font_tiny, fg=color,
                     bg=PANEL_COLOR).pack(side="left", padx=(0, 4))
            tk.Label(row, text=name, font=self.font_tiny,
                     fg=TEXT_COLOR, bg=PANEL_COLOR).pack(side="left")

            # Truncate long descriptions
            desc = info.get("desc", "")
            if len(desc) > 35:
                desc = desc[:32] + "..."
            tk.Label(row, text=desc, font=self.font_tiny,
                     fg=color, bg=PANEL_COLOR).pack(side="right")

    def log_agent_activity(self, agent_name: str, message: str, level: str = "info"):
        """Log agent activity to the live agent log panel. Thread-safe."""
        def _do():
            if not hasattr(self, "_agent_log_text"):
                return
            self._agent_log_text.config(state="normal")
            ts = datetime.now().strftime("%H:%M:%S")
            self._agent_log_text.insert("end", f"[{ts}] ", "agent_time")
            self._agent_log_text.insert("end", f"{agent_name}: ", "agent_name")
            tag = {"ok": "agent_ok", "error": "agent_err"}.get(level, "agent_info")
            self._agent_log_text.insert("end", f"{message}\n", tag)
            self._agent_log_text.see("end")
            self._agent_log_text.config(state="disabled")
        self.root.after(0, _do)

    def _update_agents_periodic(self):
        """Periodically poll AgentManager for live agent progress."""
        def _poll():
            try:
                from agent.agent_manager import AgentManager
                mgr = AgentManager()
                agents = mgr.list_agents()
                for a in agents:
                    name = a["name"]
                    status = a["status"]
                    desc = a.get("task", "")[:40]
                    result_preview = a.get("result", "")[:40]
                    if status == "completed" and result_preview:
                        desc = result_preview
                    elif status == "failed":
                        desc = a.get("error", "Error")[:40]
                    self._user_agents[name] = {
                        "status": status,
                        "desc": desc,
                        "updated": time.time(),
                    }
                # Refresh UI
                if hasattr(self, "_user_agents_frame"):
                    self._refresh_user_agents()
            except Exception:
                pass

        def _schedule():
            if not self.root.winfo_exists():
                return
            threading.Thread(target=_poll, daemon=True).start()
            self.root.after(3000, _schedule)  # Poll every 3 seconds

        # Run immediately on startup, then every 3s
        self.root.after(500, lambda: threading.Thread(target=_poll, daemon=True).start())
        self.root.after(3000, _schedule)

    # ══════════════════════════════════════════════════════════════════════════
    #  ACTIVE TOOLS
    # ══════════════════════════════════════════════════════════════════════════

    # Tool → system module mapping
    _TOOL_MODULE_MAP = {
        "browser_control": "Browser",
        "system_status": "System", "computer_settings": "System",
        "computer_control": "System", "window_manager": "System",
        "process_manager": "System",
        "notes": "Memory", "reminder": "Memory", "task_manager": "Memory",
    }

    def set_active_tool(self, tool_name: str):
        """Mark a tool as currently executing. Thread-safe."""
        if tool_name not in self._active_tools:
            self._active_tools.append(tool_name)
        self.log_activity(f"Executing: {tool_name}", "tool")
        self.set_subagent_status("Tools", "active", f"Running {tool_name}")

        # Activate related system module
        mod = self._TOOL_MODULE_MAP.get(tool_name)
        if mod:
            self.set_subagent_status(mod, "active", f"Running {tool_name}")

        # Check if this is a dynamic skill — light up the bulb
        try:
            from skills.loader import is_dynamic_skill
            if is_dynamic_skill(tool_name):
                self._active_skills.add(tool_name)
                self.set_subagent_status("Skills", "active", f"Running {tool_name}")
                self._refresh_skills_display()
        except Exception:
            pass

        # If skill_manager is running, mark Skills module active
        if tool_name == "skill_manager":
            self.set_subagent_status("Skills", "active", "Managing skills")

        self._refresh_tools_display()

    def clear_active_tool(self, tool_name: str):
        """Mark a tool as finished. Thread-safe."""
        if tool_name in self._active_tools:
            self._active_tools.remove(tool_name)
        if not self._active_tools:
            self.set_subagent_status("Tools", "idle", "Tool execution engine")

        # Deactivate related system module if no other tools need it
        mod = self._TOOL_MODULE_MAP.get(tool_name)
        if mod:
            remaining = [t for t in self._active_tools if self._TOOL_MODULE_MAP.get(t) == mod]
            if not remaining:
                defaults = {
                    "Browser": "Web browser control",
                    "System": "System monitor & control",
                    "Memory": "Long-term memory & recall",
                }
                self.set_subagent_status(mod, "idle", defaults.get(mod, mod))

        # Clear skill bulb
        if tool_name in self._active_skills:
            self._active_skills.discard(tool_name)
            self._refresh_skills_display()

        # Deactivate Skills module if no skills running
        is_skill_tool = tool_name in self._active_skills
        skill_related = tool_name == "skill_manager" or is_skill_tool
        if skill_related:
            still_running = any(
                t == "skill_manager" or t in self._active_skills
                for t in self._active_tools
            )
            if not still_running:
                from skills.loader import get_loaded_skills
                loaded = get_loaded_skills()
                desc = f"{len(loaded)} skill(s) loaded" if loaded else "No skills learned yet"
                status = "active" if loaded else "idle"
                self.set_subagent_status("Skills", status, desc)

        # Refresh the skills list after skill_manager finishes (may have added/removed)
        if tool_name == "skill_manager":
            self._load_skills_list()

        self._refresh_tools_display()

    def _refresh_tools_display(self):
        def _do():
            if not hasattr(self, "_tools_inner"):
                return
            for w in self._tools_inner.winfo_children():
                w.destroy()
            if not self._active_tools:
                tk.Label(self._tools_inner, text="No tools active",
                         font=self.font_tiny, fg=TEXT_DIM,
                         bg=PANEL_COLOR).pack(anchor="w")
                return
            for tool in self._active_tools:
                row = tk.Frame(self._tools_inner, bg=PANEL_COLOR)
                row.pack(fill="x", pady=1)
                tk.Label(row, text="\u26a1", font=self.font_tiny,
                         fg=STATUS_SPEAKING, bg=PANEL_COLOR).pack(
                    side="left", padx=(0, 4))
                tk.Label(row, text=tool, font=self.font_tiny,
                         fg=STATUS_SPEAKING, bg=PANEL_COLOR,
                         anchor="w").pack(side="left")

        self.root.after(0, _do)

    # ══════════════════════════════════════════════════════════════════════════
    #  LEARNED SKILLS PANEL
    # ══════════════════════════════════════════════════════════════════════════

    def _load_skills_list(self):
        """Load dynamic skills list in a background thread."""
        def _work():
            try:
                from skills.loader import get_loaded_skills
                skills = get_loaded_skills()
                info = []
                for name, mod in sorted(skills.items()):
                    desc = getattr(mod, "TOOL_DESC", "")[:45]
                    info.append({"name": name, "desc": desc})
                self._loaded_skills_info = info
                self.root.after(0, self._refresh_skills_display)
            except Exception:
                pass

        threading.Thread(target=_work, daemon=True).start()

    def refresh_skills(self):
        """Public method to reload the skills list display. Thread-safe."""
        self._load_skills_list()

    def _refresh_skills_display(self):
        """Redraw the learned skills panel with bulb indicators."""
        def _do():
            if not hasattr(self, "_skills_inner"):
                return
            for w in self._skills_inner.winfo_children():
                w.destroy()
            if not self._loaded_skills_info:
                tk.Label(self._skills_inner, text="No skills learned yet",
                         font=self.font_tiny, fg=TEXT_DIM,
                         bg=PANEL_COLOR).pack(anchor="w")
                return
            for skill in self._loaded_skills_info:
                name = skill["name"]
                desc = skill["desc"]
                is_active = name in self._active_skills

                row = tk.Frame(self._skills_inner, bg=PANEL_COLOR)
                row.pack(fill="x", pady=1)

                # Light bulb icon — bright yellow when active, dim when idle
                bulb_color = SKILL_BULB_ON if is_active else SKILL_BULB_OFF
                bulb = tk.Label(row, text="\U0001f4a1", font=self.font_tiny,
                                fg=bulb_color, bg=PANEL_COLOR)
                bulb.pack(side="left", padx=(0, 4))

                # Glow effect on the name when active
                name_color = SKILL_BULB_GLOW if is_active else TEXT_COLOR
                tk.Label(row, text=name, font=self.font_tiny,
                         fg=name_color, bg=PANEL_COLOR).pack(side="left")

                # Description on the right
                if desc and len(desc) > 25:
                    desc = desc[:22] + "..."
                tk.Label(row, text=desc, font=self.font_tiny,
                         fg=TEXT_DIM if not is_active else SKILL_BULB_ON,
                         bg=PANEL_COLOR).pack(side="right")

        self.root.after(0, _do)

    # ══════════════════════════════════════════════════════════════════════════
    #  INPUT HANDLING
    # ══════════════════════════════════════════════════════════════════════════

    def _clear_placeholder(self, event):
        if self._text_input.get() == "Type a command...":
            self._text_input.delete(0, "end")
            self._text_input.config(fg=TEXT_COLOR)

    def _restore_placeholder(self, event):
        if not self._text_input.get().strip():
            self._text_input.delete(0, "end")
            self._text_input.insert(0, "Type a command...")
            self._text_input.config(fg=TEXT_DIM)

    def _handle_text_input(self, event):
        text = self._text_input.get().strip()
        if not text or text == "Type a command...":
            return
        self._text_input.delete(0, "end")
        self._append_log("user", text)
        self.log_activity(
            f"User: {text[:50]}{'...' if len(text) > 50 else ''}", "info")
        if self._on_text_input:
            self._on_text_input(text)

    # ══════════════════════════════════════════════════════════════════════════
    #  CONVERSATION LOG
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _has_hebrew(text: str) -> bool:
        return any("\u0590" <= ch <= "\u05FF" for ch in text)

    @staticmethod
    def _lrm_wrap(text: str) -> str:
        return _LTR_TOKEN_RE.sub(lambda m: f"{_LRM}{m.group(0)}{_LRM}", text)

    def _append_log(self, role: str, text: str, typewriter: bool = False):
        """Add a complete message to the conversation log. Thread-safe."""
        if not hasattr(self, "_log_text"):
            return

        def _do():
            self._log_text.config(state="normal")
            prefix = {"user": "You: ", "jarvis": "JARVIS: ",
                      "system": ">>> ", "tool_msg": "[Tool] "}.get(role, "")
            is_rtl = self._has_hebrew(text)
            clean = text.rstrip("\n")
            if is_rtl:
                msg = self._lrm_wrap(clean)
                line = (f"{_RLM}{_LRM}{prefix}{_LRM}{msg}\n" if prefix
                        else f"{_RLM}{msg}\n")
                tag = ("user_rtl" if role == "user"
                       else "jarvis_rtl" if role == "jarvis"
                       else role)
            else:
                line = prefix + clean + "\n"
                tag = role
            self._log_text.insert("end", line, tag)
            self._log_text.see("end")
            self._log_text.config(state="disabled")

        self.root.after(0, _do)

    def append_token(self, token: str):
        self._jarvis_buffer += token

    def append_user_token(self, token: str):
        self._user_buffer += token

    def finish_jarvis_line(self):
        text = self._jarvis_buffer.strip()
        self._jarvis_buffer = ""
        if text:
            self._append_log("jarvis", text)
            self.log_activity(
                f"JARVIS: {text[:50]}{'...' if len(text) > 50 else ''}", "system")

    def finish_user_line(self):
        text = self._user_buffer.strip()
        self._user_buffer = ""
        if text:
            self._append_log("user", text)
            self.log_activity(
                f"User (voice): {text[:40]}{'...' if len(text) > 40 else ''}", "info")

    # ══════════════════════════════════════════════════════════════════════════
    #  STATUS
    # ══════════════════════════════════════════════════════════════════════════

    def set_status(self, status: str):
        """Update the status indicator. Thread-safe."""
        old = self._status
        self._status = status
        colors = {
            "ONLINE": STATUS_ONLINE, "LISTENING": STATUS_ONLINE,
            "SPEAKING": STATUS_SPEAKING, "PROCESSING": STATUS_PROCESS,
            "CONNECTING": STATUS_PROCESS, "RECONNECTING": STATUS_PROCESS,
            "AUTH_ERROR": "#ff5252", "OFFLINE": "#ff5252",
        }
        color = colors.get(status, ACCENT_DIM)

        def _do():
            if hasattr(self, "_status_label"):
                self._status_label.config(text=status, fg=color)
                self._status_dot.delete("all")
                self._status_dot.create_oval(1, 1, 9, 9, fill=color, outline="")

        self.root.after(0, _do)

        if status == "LISTENING":
            self.set_subagent_status("Voice", "active", "Listening for speech")
        elif status == "SPEAKING":
            self.set_subagent_status("Voice", "active", "Playing audio response")
        elif status in ("ONLINE", "OFFLINE"):
            self.set_subagent_status("Voice", "idle", "Speech I/O pipeline")

        if old != status:
            self.log_activity(f"Status \u2192 {status}", "system")

    def set_audio_level(self, level: float):
        self._audio_level = min(level * 5, 1.0)

    # ══════════════════════════════════════════════════════════════════════════
    #  WAVEFORM
    # ══════════════════════════════════════════════════════════════════════════

    def _animate_waveform(self):
        if not hasattr(self, "_wave_canvas") or not self._wave_canvas.winfo_exists():
            return
        canvas = self._wave_canvas
        w = canvas.winfo_width() or 600
        h = canvas.winfo_height() or 55
        canvas.delete("all")

        self._waveform_data.pop(0)
        if self._status in ("SPEAKING", "PROCESSING"):
            val = self._audio_level + random.uniform(0.05, 0.3)
        elif self._status == "LISTENING":
            val = self._audio_level * 0.5 + random.uniform(0.01, 0.05)
        else:
            val = random.uniform(0.01, 0.03)
        self._waveform_data.append(min(val, 1.0))

        bar_count = len(self._waveform_data)
        bar_w = max(w / bar_count - 2, 2)
        mid_y = h / 2

        for i, v in enumerate(self._waveform_data):
            x = i * (w / bar_count)
            bar_h = max(v * mid_y * 0.85, 1)
            intensity = min(int(v * 255), 255)
            color = f"#{0:02x}{min(180 + intensity // 3, 255):02x}{min(216 + intensity // 6, 255):02x}"
            canvas.create_rectangle(x, mid_y - bar_h, x + bar_w, mid_y + bar_h,
                                    fill=color, outline="")

        canvas.create_line(0, mid_y, w, mid_y, fill=ACCENT_DIM, width=1, dash=(3, 3))
        self.root.after(33, self._animate_waveform)

    # ══════════════════════════════════════════════════════════════════════════
    #  API KEY PERSISTENCE
    # ══════════════════════════════════════════════════════════════════════════

    def _poll_skills_periodic(self):
        """Periodically refresh the skills list (new skills may be created mid-session)."""
        self._load_skills_list()
        self.root.after(15000, self._poll_skills_periodic)  # every 15s

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
        return key.startswith("AIza") and len(key) > 20

    def show_api_key_screen(self, error_msg: str = ""):
        def _do():
            self._api_key = None
            self._build_api_key_screen()
            if error_msg and hasattr(self, "_key_error"):
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

    # ══════════════════════════════════════════════════════════════════════════
    #  HELPERS
    # ══════════════════════════════════════════════════════════════════════════

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
