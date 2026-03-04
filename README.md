<div align="center">

# 🤖 J.A.R.V.I.S

**Just A Rather Very Intelligent System**

A real-time AI voice assistant powered by Gemini Live — with tool control, forever memory, and Iron Man HUD.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://python.org)
[![Gemini](https://img.shields.io/badge/Gemini_2.5_Flash-Native_Audio-4285F4.svg)](https://ai.google.dev)

</div>

---

## What is JARVIS?

JARVIS is a voice-controlled AI assistant inspired by Tony Stark's companion — with **Paul Bettany's calm, dry British wit**. It uses Google's Gemini Live API for real-time bidirectional audio conversation.

### Key Features

- **Real-time Voice** — Gemini 2.5 Flash native audio. Talk naturally, JARVIS responds instantly in the Charon voice.
- **Bilingual** — English and Hebrew. JARVIS responds in the language you speak.
- **14 Tool Categories** — Launch apps, search the web, control browser (Playwright), manage files, run commands, control mouse/keyboard, adjust system settings (volume, brightness, dark mode), write/run code, YouTube, reminders, screenshot analysis, and more.
- **Forever Memory** — SQLite + ChromaDB vector memory. JARVIS remembers your name, preferences, past conversations, and learns from every interaction.
- **Iron Man HUD** — Tkinter GUI with animated waveform visualizer, conversation log, and status indicators.
- **Pure Python** — Everything runs in a single `python main.py`. No Node.js, no Docker, no complex infrastructure.

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/yourusername/jarvis.git
cd jarvis

# 2. Setup (installs deps + Playwright browsers)
python setup.py

# 3. Run
python main.py
```

On first launch, JARVIS will ask for your **Gemini API key** (get one free at [ai.google.dev](https://ai.google.dev)).

---

## Architecture

```
main.py              → Entry point: GUI + Gemini Live on daemon thread
gemini_live.py       → Core audio engine (4 async tasks: listen/send/receive/play)
gui.py               → Tkinter Iron Man HUD (waveform, log, status, text input)
tool_registry.py     → 14 tool declarations (Gemini format) + dispatcher
agent/personality.py → JARVIS personality prompt (Paul Bettany style)
memory/              → Forever memory (SQLite facts + ChromaDB vectors)
tools/               → Browser control (Playwright), system monitor, computer control
```

### How it works

1. **PyAudio** captures your microphone at 16kHz
2. Audio streams to **Gemini Live** via WebSocket
3. Gemini responds with audio (24kHz) + text transcription + tool calls
4. Tool calls are dispatched to local Python functions
5. Audio plays through speakers, text appears in GUI
6. Conversations are saved to **forever memory** (SQLite + ChromaDB)

---

## Tools

| Tool | Description |
|------|-------------|
| `open_app` | Launch any application (Chrome, Notepad, Spotify, VS Code, etc.) |
| `web_search` | Search the web via DuckDuckGo |
| `weather_report` | Current weather for any city (Open-Meteo) |
| `get_current_time` | Time in any timezone |
| `system_status` | CPU, RAM, disk usage |
| `computer_settings` | Volume, brightness, window mgmt, dark mode, WiFi, hotkeys |
| `browser_control` | Navigate, search, click, type, fill forms (Playwright) |
| `file_controller` | List, create, delete, move, copy, read, write files |
| `cmd_control` | Execute system commands |
| `code_helper` | Write, edit, run, explain code |
| `youtube_video` | Play videos, show trending |
| `reminder` | Set timed reminders |
| `screen_process` | Screenshot + vision analysis |
| `computer_control` | Direct mouse/keyboard (click, type, drag, hotkey) |

---

## Requirements

- **Python 3.10+**
- **Windows** (primary target; macOS/Linux may need tweaks for some tools)
- **Gemini API key** (free tier available at [ai.google.dev](https://ai.google.dev))
- **Microphone + Speakers**

### Dependencies (15 packages)

```
google-genai, pyaudio, pyautogui, pynput, keyboard, mouse,
chromadb, numpy, loguru, PyYAML, psutil, httpx, Pillow, playwright
```

---

## Project Structure

```
jarvis/
├── main.py              # Entry point
├── gemini_live.py       # Gemini Live audio engine
├── gui.py               # Tkinter HUD
├── tool_registry.py     # Tool declarations + dispatcher
├── setup.py             # One-command installer
├── requirements.txt     # Dependencies
├── agent/
│   ├── personality.py   # JARVIS character prompt
│   ├── timing_context.py
│   └── utils.py
├── memory/
│   ├── long_term.py     # Long-term memory API
│   ├── manager.py       # Memory manager (SQLite + ChromaDB)
│   ├── sqlite_store.py  # SQLite storage
│   ├── vector_store.py  # ChromaDB vector search
│   └── short_term.py    # Session memory
├── tools/
│   ├── browser_control.py  # Playwright automation
│   ├── computer_control.py # PyAutoGUI actions
│   └── system_monitor.py   # System stats
├── config/              # API keys, settings
├── data/                # ChromaDB, SQLite (gitignored)
└── log/                 # Daily logs (gitignored)
```

---

## License

MIT
