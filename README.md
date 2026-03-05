<div align="center">

# 🤖 J.A.R.V.I.S

**Just A Rather Very Intelligent System**

A real-time AI voice assistant powered by Google Gemini Live — with 28 tools, forever memory, autonomous agents, and an Iron Man HUD.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://python.org)
[![Gemini 2.5 Flash](https://img.shields.io/badge/Gemini_2.5_Flash-Native_Audio-4285F4.svg)](https://ai.google.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub Sponsors](https://img.shields.io/badge/Sponsor-%E2%9D%A4-red?logo=github)](https://github.com/sponsors/adirangel)

<br />

**Talk to your computer like Tony Stark talks to JARVIS.**

JARVIS listens through your microphone, thinks with Gemini, controls your PC with 28 tools, remembers everything forever, and responds with Paul Bettany's calm British wit — all in a single `python main.py`.

</div>

---

## Table of Contents

- [What is JARVIS?](#what-is-jarvis)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [The Iron Man HUD](#the-iron-man-hud)
- [Tools (28)](#tools-28)
- [Memory System](#memory-system)
- [Agent System](#agent-system)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [Sponsor](#sponsor)
- [License](#license)

---

## What is JARVIS?

JARVIS is a **voice-controlled AI assistant** inspired by Tony Stark's companion from the Iron Man films. It uses Google's **Gemini 2.5 Flash Native Audio** for real-time bidirectional voice conversation over WebSocket — you talk, JARVIS listens, thinks, acts, and responds instantly with voice.

### Key Features

| Feature | Description |
|---------|-------------|
| **Real-Time Voice** | Gemini 2.5 Flash native audio — talk naturally, JARVIS responds instantly in the "Charon" voice |
| **Bilingual** | English and Hebrew — JARVIS auto-detects and responds in the language you speak |
| **28 Tools** | Launch apps, browse the web, control mouse/keyboard, manage files, run code, set reminders, and much more |
| **Forever Memory** | SQLite + ChromaDB vector store — JARVIS remembers your name, preferences, and past conversations across sessions |
| **Autonomous Agents** | Spawn background agents for long-running tasks (monitoring, research, multi-step workflows) |
| **Iron Man HUD** | Tkinter GUI with animated waveform, system gauges, conversation log, Mission Control panel |
| **Pure Python** | Single command to run — no Node.js, no Docker, no complex infrastructure |

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/adirangel/JARVIS.git
cd JARVIS
```

### 2. Run setup (installs all dependencies + Playwright browsers)

```bash
python setup.py
```

### 3. Launch JARVIS

```bash
python main.py
```

On first launch, JARVIS will display an API key entry screen. Enter your **Gemini API key** (get one free at [ai.google.dev](https://ai.google.dev)). The key is saved locally in `config/api_keys.json` and never leaves your machine.

That's it — start talking!

---

## How It Works

JARVIS runs entirely on your local machine, with Google Gemini handling the AI processing in the cloud. Here's the complete flow from your voice to JARVIS's response:

```
                         YOUR COMPUTER                              CLOUD
                    ┌─────────────────────────────┐          ┌──────────────┐
                    │                             │          │              │
  🎤 You speak ──► │  PyAudio captures mic       │          │              │
                    │  (16kHz audio stream)       │          │              │
                    │         │                   │          │              │
                    │         ▼                   │          │              │
                    │  Send audio via WebSocket ──┼────────► │  Gemini 2.5  │
                    │                             │          │  Flash Live  │
                    │                             │          │  (Native     │
                    │  Receive response ◄─────────┼──────────│   Audio)     │
                    │    ├── Audio (24kHz voice)   │          │              │
                    │    ├── Text transcription    │          │              │
                    │    └── Tool calls            │          │              │
                    │         │                   │          └──────────────┘
                    │         ▼                   │
                    │  ┌─ Audio → Speakers 🔊     │
                    │  ├─ Text  → GUI log         │
                    │  └─ Tools → Local execution │
                    │         │                   │
                    │         ▼                   │
                    │  Save to Memory             │
                    │  (SQLite + ChromaDB)         │
                    └─────────────────────────────┘
```

### Step by Step

1. **Capture** — PyAudio captures your microphone input at 16kHz in real-time
2. **Stream** — Audio is streamed to Gemini 2.5 Flash Live API via WebSocket (with echo suppression to prevent JARVIS from hearing itself)
3. **Process** — Gemini processes your speech and returns three things simultaneously:
   - **Audio response** (24kHz) — JARVIS's voice reply
   - **Text transcription** — what you said and what JARVIS says (displayed in GUI)
   - **Tool calls** — if JARVIS needs to take action (open an app, search the web, etc.)
4. **Execute tools** — Tool calls are dispatched to local Python functions that control your PC
5. **Play audio** — JARVIS's voice response plays through your speakers
6. **Save memory** — The conversation is saved to SQLite + ChromaDB for long-term recall

### Threading Model

JARVIS uses two threads:
- **Main thread** — Tkinter GUI (Iron Man HUD)
- **Daemon thread** — `GeminiLive` engine with 4 concurrent async tasks:
  - `_listen_audio` — captures mic input
  - `_send_realtime` — streams audio to Gemini
  - `_receive_audio` — handles Gemini's audio/text/tool responses
  - `_play_audio` — plays JARVIS's voice through speakers

---

## The Iron Man HUD

JARVIS comes with a full Tkinter-based Iron Man HUD with a 3-column layout:

| Panel | Contents |
|-------|----------|
| **Left — System Specs** | CPU, RAM, Disk, GPU, Network gauges with color coding (green → yellow → red) + uptime counter |
| **Center — Main View** | Animated waveform/audio visualizer, JARVIS status indicator (ONLINE / LISTENING / SPEAKING / PROCESSING), scrolling conversation log with Hebrew BiDi support |
| **Right — Mission Control** | Activity log, currently active tools, tasks & reminders list, sub-agent status with live progress |
| **Bottom** | Text input bar for typed commands (when you can't talk) |

Color scheme: dark background (`#060a13`) with cyan/electric blue accents (`#00e5ff`), matching the Iron Man aesthetic.

On first run, an animated arc-reactor API key entry screen appears.

---

## Tools (28)

JARVIS can control your computer through 28 built-in tools. When you ask JARVIS to do something, Gemini automatically decides which tool to call.

### Computer & Apps (6)

| Tool | What It Does |
|------|-------------|
| `open_app` | Launch any application — Chrome, Spotify, VS Code, Notepad, etc. |
| `computer_settings` | Volume, brightness, minimize/maximize windows, scroll, zoom, screenshot, lock, restart, shutdown, hotkeys, dark mode, WiFi toggle |
| `computer_control` | Direct mouse & keyboard control — click, type, drag, press keys, hotkeys, scroll |
| `window_manager` | List, switch, snap, minimize, or close windows |
| `process_manager` | List running processes, kill processes, check if an app is running |
| `clipboard_manager` | Read from or write to the system clipboard |

### Web & Information (4)

| Tool | What It Does |
|------|-------------|
| `web_search` | Search the web via DuckDuckGo — returns titles + snippets |
| `browser_control` | Full browser automation using your real Chrome/Edge — navigate, search, click, type, fill forms, manage tabs |
| `weather_report` | Current weather for any city (Open-Meteo API) — temperature, humidity, wind |
| `news` | Get latest news headlines by topic |

### Productivity (6)

| Tool | What It Does |
|------|-------------|
| `reminder` | Set, list, and complete persistent reminders (saved to SQLite) |
| `task_manager` | Add, list, complete, and delete tasks (saved to SQLite) |
| `notes` | Save, list, search, and delete persistent notes |
| `timer` | Set countdown timers with notification |
| `daily_briefing` | Morning report — time, weather, system stats, tasks, reminders, news |
| `get_current_time` | Current date/time in any timezone (maps city names automatically) |

### Files & Code (3)

| Tool | What It Does |
|------|-------------|
| `file_controller` | List, create, delete, move, copy, rename, read, write, search files + create directories |
| `code_helper` | Write, edit, read, explain, and run code files (with visible terminal output) |
| `cmd_control` | Execute any shell command (CMD/PowerShell), optionally in a visible terminal |

### Media & Utilities (3)

| Tool | What It Does |
|------|-------------|
| `youtube_video` | Play YouTube videos by search query or show trending videos |
| `screen_process` | Take a screenshot and analyze what's on screen using vision |
| `translate` | Multi-language translation (Hebrew, English, and more) |

### System (1)

| Tool | What It Does |
|------|-------------|
| `system_status` | CPU, RAM, and disk usage — formatted for speech |

### Agent Management (6)

| Tool | What It Does |
|------|-------------|
| `spawn_agent` | Create a background agent (types: command, script, monitor, research, tool, multi-step) |
| `agent_status` | Check the status of one or all running agents |
| `agent_result` | Get the full result from a completed agent |
| `stop_agent` | Stop a running agent |
| `agent_message` | Send messages between agents or broadcast to all |
| `remove_agent` | Remove a stopped or completed agent from the registry |

---

## Memory System

JARVIS has a **forever memory** — it remembers your name, preferences, and past conversations across sessions. Memory is stored locally on your machine and never shared.

### How Memory Works

```
Conversation happens
        │
        ▼
┌─────────────────────┐
│  Every 3rd turn:    │
│  LLM Fact Extraction│
│                     │
│  Stage 1 (cheap):   │
│  "Does this contain │
│   personal info?"   │
│   → YES / NO        │
│                     │
│  Stage 2 (if YES):  │
│  Extract facts as   │
│  {key, value,       │
│   confidence} JSON  │
└─────────┬───────────┘
          │
     ┌────┴────┐
     ▼         ▼
  SQLite    ChromaDB
  (facts,   (vector
   tasks,    embeddings
   profile)  for semantic
             search)
```

### Storage Layers

| Layer | Technology | What It Stores |
|-------|-----------|----------------|
| **Facts** | SQLite | Key-value facts about you (name, preferences, etc.) with confidence scoring |
| **Messages** | SQLite | Full conversation history with timestamps |
| **User Profile** | SQLite | Personal info with confidence boosting — repeated mentions increase confidence |
| **Tasks & Reminders** | SQLite | Persistent to-do items and timed reminders |
| **Vector Search** | ChromaDB | Embedded conversation chunks for semantic search — find relevant past conversations by meaning |

### Relevance Scoring

When JARVIS recalls memories, results are ranked using:

$$\text{score} = \text{cosine similarity} \times (1 + 0.3 \times \text{recency}) \times (1 + 0.1 \times \text{access count})$$

Where recency = $e^{-0.01 \times \text{days old}}$ (half-life ~70 days). Recent and frequently accessed memories rank higher.

### Memory Consolidation

Old conversations are automatically consolidated:
- **Daily summaries** — group messages by day, LLM summarizes key facts
- **Weekly/Monthly** — further consolidation over time
- Summaries are embedded into ChromaDB for future semantic search

---

## Agent System

JARVIS can spawn **autonomous background agents** that run in separate threads. This is useful for tasks that take time or need ongoing monitoring.

### Agent Types

| Type | Description | Example |
|------|-------------|---------|
| `command` | Run a single shell command | "Run a disk cleanup" |
| `script` | Execute a Python/batch script | "Run my backup script" |
| `monitor` | Periodically check something | "Monitor CPU and alert if > 90%" |
| `research` | Multi-step web research | "Research the latest Python 3.13 features" |
| `tool` | Call a JARVIS tool | "Download this file in the background" |
| `multi_step` | Execute a sequence of steps | "Create a project folder, init git, and add a README" |

Agents report results back to JARVIS, which can then relay them to you. Agent state is persisted in SQLite, so they survive restarts.

---

## Project Structure

```
JARVIS/
├── main.py               # Entry point — GUI + Gemini Live on daemon thread
├── gemini_live.py         # Core engine — 4 async tasks over WebSocket
├── gui.py                 # Tkinter Iron Man HUD (3-column layout)
├── tool_registry.py       # 28 tool declarations + dispatcher
├── setup.py               # One-command installer
├── requirements.txt       # Python dependencies
│
├── agent/
│   ├── agent_manager.py   # AgentManager — spawn/manage background agents
│   ├── personality.py     # JARVIS personality prompt (Paul Bettany style)
│   └── utils.py           # Logging and debug utilities
│
├── memory/
│   ├── long_term.py       # Long-term memory API
│   ├── manager.py         # Memory manager (2-stage fact extraction + consolidation)
│   ├── sqlite_store.py    # SQLite storage (10 tables: facts, messages, profile, tasks...)
│   ├── vector_store.py    # ChromaDB vector search (cosine + recency + access boosting)
│   └── short_term.py      # In-memory session context
│
├── tools/
│   ├── browser_control.py # Browser automation (real Chrome/Edge via keyboard/mouse)
│   ├── computer_control.py# Mouse & keyboard control (pynput/pyautogui)
│   └── system_monitor.py  # System stats (psutil)
│
├── skills/                # Extensible skill framework (for future plugins)
│   └── base.py            # Skill contract definition
│
├── config/                # API keys (gitignored)
├── data/                  # SQLite + ChromaDB data (gitignored)
└── log/                   # Daily logs with 7-day rotation (gitignored)
```

---

## Configuration

### API Key

On first launch, JARVIS prompts for your Gemini API key through the GUI. The key is stored locally in `config/api_keys.json` (gitignored — never committed).

You can get a free Gemini API key at [ai.google.dev](https://ai.google.dev).

### Config Files

| File | Purpose |
|------|---------|
| `config.example.yaml` | Reference template with all available settings |
| `config.yaml` | Your local configuration (optional — JARVIS works with defaults) |

The config supports settings for LLM provider, voice parameters, wake word sensitivity, memory paths, tool restrictions, and debug options. For most users, the defaults work out of the box.

---

## Requirements

| Requirement | Details |
|-------------|---------|
| **Python** | 3.10 or higher |
| **OS** | Windows (primary target — macOS/Linux may need tweaks for some tools) |
| **API Key** | Gemini API key (free tier available at [ai.google.dev](https://ai.google.dev)) |
| **Hardware** | Microphone + Speakers (or headset) |

### Python Dependencies

All installed automatically by `python setup.py`:

```
google-genai    pyaudio       pyautogui     pynput        keyboard
mouse           chromadb      numpy         loguru        PyYAML
psutil          httpx         Pillow        python-bidi
```

---

## Contributing

Contributions are welcome! Here's how to get started:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Test locally with `python main.py`
5. Commit and push (`git push origin feature/my-feature`)
6. Open a Pull Request

### Ideas for Contributions

- New tools (Spotify control, calendar integration, email, etc.)
- macOS / Linux compatibility fixes
- New GUI themes or layouts
- Skills framework plugins
- Test coverage for the current architecture

---

## Sponsor

If you find JARVIS useful, consider sponsoring the project to support development:

<div align="center">

[![Sponsor](https://img.shields.io/badge/%E2%9D%A4%EF%B8%8F_Sponsor-JARVIS-red?style=for-the-badge&logo=github)](https://github.com/sponsors/adirangel)

</div>

Your support helps keep JARVIS actively maintained and growing with new features.

---

## License

MIT — see [LICENSE](LICENSE) for details.
