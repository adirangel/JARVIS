<div align="center">

# 🤖 J.A.R.V.I.S

**Just A Rather Very Intelligent System**

A 100% local AI assistant with voice interaction, wake word detection, and a web dashboard.
Powered by Ollama, with Paul Bettany's JARVIS personality.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://python.org)
[![Next.js 14](https://img.shields.io/badge/Next.js-14-000000.svg)](https://nextjs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688.svg)](https://fastapi.tiangolo.com)

<a href="https://github.com/sponsors/adirangel">
  <img src="https://img.shields.io/badge/%E2%9D%A4%EF%B8%8F_Sponsor-JARVIS-ea4aaa?style=for-the-badge&logo=github-sponsors" alt="Sponsor JARVIS" />
</a>

</div>

---

## 🧠 What is JARVIS?

Inspired by Tony Stark's AI companion from the Marvel universe — voiced by Paul Bettany — **JARVIS** is your personal, always-ready assistant that lives entirely on your machine. No cloud. No subscriptions. No data leaving your PC.

Unlike cloud-based assistants (Alexa, Siri, ChatGPT), JARVIS is:

- **Private** — Everything runs locally. Your conversations, your data, your hardware. Nothing is sent to any server.
- **Autonomous** — JARVIS doesn't just answer questions. He plans, reflects on his own responses, uses tools (web search, system monitoring, computer control), and remembers past conversations.
- **Conversational** — Say "Hey Jarvis" once, then talk naturally. He listens continuously, responds with a streaming voice (Piper TTS), and waits for you — no need to repeat the wake word every time.
- **Extensible** — Add new tools, swap LLM models, tweak his personality, or build skills on top of the plugin system.

### Our Goal

Build a **fully local, open-source AI assistant** that rivals commercial voice assistants in responsiveness and usefulness — while keeping 100% of your data on your own hardware. We target **sub-2-second end-to-end latency** (wake → STT → LLM → TTS) on consumer GPUs, and aim to make JARVIS the best starting point for anyone who wants their own private AI companion.

---

## ✨ Features

- **Voice Assistant** — Wake word ("Hey Jarvis"), continuous conversation, streaming TTS
- **100% Local** — Runs entirely on your machine via Ollama (no cloud APIs required)
- **Web Dashboard** — Real-time system stats, weather, conversation panel, audio visualizer
- **Agent Architecture** — Input → Reflector → Tool → Output node graph with planning
- **Tool Use** — Web search (DuckDuckGo), system monitoring, computer control, time/date
- **Memory** — Short-term conversation + long-term ChromaDB vector memory
- **Streaming TTS** — Piper TTS with sentence-boundary streaming for low latency
- **Personality** — Configurable personality system (default: Paul Bettany's JARVIS)

---

## 📋 Requirements

| Item | Details |
|------|---------|
| **OS** | Windows 10/11 |
| **Python** | 3.10 or newer |
| **Node.js** | 18+ (for the web dashboard) |
| **Ollama** | Running locally ([ollama.com](https://ollama.com)) |
| **GPU** *(optional)* | NVIDIA with CUDA 12 — speeds up speech-to-text |
| **Microphone** | Required for voice mode |

---

## 🚀 Quick Start

### 1. Clone the repository

```powershell
git clone <repo-url> jarvis
cd jarvis
```

### 2. Set up Python environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Install Ollama and pull models

Download and install [Ollama](https://ollama.com), then pull the required models:

```powershell
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

> `llama3.2:3b` is used for all LLM tasks (conversation, planning, reflection, tools).
> `nomic-embed-text` is used for long-term memory embeddings (optional).

### 4. Configure JARVIS

JARVIS reads its configuration from `config/settings.yaml`. A root-level `config.yaml` is also available. Review and edit as needed:

```powershell
notepad config/settings.yaml
```

Key settings to check:
- **`input_device`** — Microphone device ID (run `python -m sounddevice` to list devices)
- **`llm_provider`** / **`llm_model`** — LLM backend (`ollama` + `llama3.2:3b` by default)
- **`tts_engine`** — Text-to-speech engine (`piper` by default)

### 5. Install the dashboard UI

```powershell
cd ui
npm install
cd ..
```

---

## 🏃 Running JARVIS

JARVIS has three runtime modes plus a web dashboard. You can run them independently or together.

### Option A: Web Dashboard (recommended for first run)

The dashboard gives you a visual interface with system stats, weather, conversation, and an audio visualizer.

**Terminal 1 — Start the API backend:**

```powershell
cd jarvis
uvicorn api.main:app --reload --port 8000
```

**Terminal 2 — Start the Next.js frontend:**

```powershell
cd jarvis/ui
npm run dev
```

Open **http://localhost:3000** in your browser. The dashboard shows:
- **Left sidebar** — System stats (CPU/RAM/Disk), weather, camera toggle, session uptime
- **Center** — Audio visualizer with voice status, backend connection indicator
- **Right panel** — Conversation with JARVIS (type messages, clear/export history)

The header displays real-time clock, weather summary, and backend connection status (Online/Offline + WebSocket indicator).

> **Note:** The dashboard communicates with the API at `http://localhost:8000`. If the backend isn't running, you'll see a red "Backend offline" banner in the center panel.

### Option B: Voice Mode (wake word)

Say **"Hey Jarvis"** to activate, then speak naturally. After the first wake, JARVIS listens continuously until ~15 seconds of silence or you say "goodbye" / "stop".

```powershell
python main.py --mode voice
```

### Option C: CLI Mode (text only)

Type messages in the terminal. Great for testing without a microphone.

```powershell
python main.py --mode cli
```

You can also pipe input for single-shot testing:

```powershell
echo "what time is it?" | python main.py --mode cli
```

### Option D: Single Test Message

Run a single message through the agent and exit:

```powershell
python main.py --mode test --test-text "What is your name?"
```

### Running Everything Together

For the full experience, run three terminals:

| Terminal | Command | Purpose |
|----------|---------|---------|
| 1 | `uvicorn api.main:app --reload --port 8000` | API backend |
| 2 | `cd ui && npm run dev` | Web dashboard |
| 3 | `python main.py --mode voice` | Voice assistant |

---

## 🔧 Configuration Reference

Configuration is in `config/settings.yaml`. Key sections:

### LLM

| Key | Default | Description |
|-----|---------|-------------|
| `llm_provider` | `ollama` | LLM backend provider |
| `llm_model` | `llama3.2:3b` | Model name |
| `llm_temperature` | `0.8` | Response creativity (0.0–1.0) |
| `llm_max_tokens` | `256` | Max response tokens |

### Voice

| Key | Default | Description |
|-----|---------|-------------|
| `input_device` | `null` | Mic device ID (null = default) |
| `tts_engine` | `piper` | TTS engine |
| `tts_speed` | `0.95` | Speech speed |

### Wake Word

| Key | Default | Description |
|-----|---------|-------------|
| `wake_word` | `Hey Jarvis` | Trigger phrase |
| `debounce_seconds` | `1.5` | Cooldown between triggers |

### Memory

| Key | Default | Description |
|-----|---------|-------------|
| `memory_long_term_enabled` | `false` | Enable ChromaDB long-term memory |
| `max_conversation_turns` | `15` | Short-term memory window |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Web Dashboard (Next.js)                │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ Sidebar  │  │   Center     │  │  Conversation    │   │
│  │ Stats    │  │  Visualizer  │  │  Panel           │   │
│  │ Weather  │  │  Status      │  │  Chat + Input    │   │
│  │ Uptime   │  │  Controls    │  │                  │   │
│  └──────────┘  └──────────────┘  └──────────────────┘   │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP + WebSocket
┌──────────────────────▼──────────────────────────────────┐
│              FastAPI Backend (port 8000)                  │
│  /api/health  /api/system  /api/weather  /api/uptime     │
│  /api/conversation (GET/POST/DELETE)  /api/voice/status  │
│  /ws (WebSocket)                                         │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                  Agent Graph                             │
│  InputNode → ReflectorNode → ToolNode → OutputNode       │
│                                                          │
│  Tools: web_search, system_monitor, computer_control,    │
│         get_time, get_date                               │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│  Ollama (llama3.2:3b)│  Memory (SQLite + ChromaDB)       │
│  Piper TTS          │  faster-whisper STT               │
└─────────────────────────────────────────────────────────┘
```

---

## 🧪 Verify Your Setup

**List available audio devices:**

```powershell
python -m sounddevice
```

Set `input_device` in `config/settings.yaml` to the device ID of your preferred microphone.

**Run unit tests:**

```powershell
python -m pytest tests/ -v
```

---

## ⚡ Latency Optimization

JARVIS targets sub-2s end-to-end latency. Key optimizations:

- **FastPath** — Simple commands (greetings, thanks) skip the Planner and go straight to the Reflector
- **Streaming TTS** — First phrase plays within ~800ms while the rest generates
- **Single model** — `llama3.2:3b` handles all tasks (no model switching overhead)
- **Quantized STT** — `faster-whisper` with `int8` compute + `beam_size=3`
- **Preloaded TTS** — Piper model loaded at startup
- **Memory cache** — ChromaDB recent query cache for <100ms lookups

Enable debug timing to see per-node latency:

```yaml
# In config/settings.yaml
debug: true
```

Timer data is logged to `log/jarvis-*.log` at DEBUG level.

---

## 🔍 Troubleshooting

### Backend won't start

```
ModuleNotFoundError: No module named 'xxx'
```

Make sure your virtual environment is activated and dependencies are installed:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Dashboard shows "Backend offline"

The API server isn't running. Start it with:

```powershell
uvicorn api.main:app --reload --port 8000
```

### False wake word triggers

JARVIS uses multi-layer wake word filtering: WebRTC VAD pre-gating, Whisper with anti-hallucination parameters, fuzzy matching with Levenshtein distance, and post-STT transcript validation. If false triggers still occur:

1. Ensure `input_device` points to a good microphone (not a silent or noisy device)
2. Increase `debounce_seconds` in config to add cooldown between triggers

### Whisper transcribes phantom words ("Thank you", etc.)

This is handled automatically by the validation layer in `voice/validation.py`. Known noise artifacts are filtered out, and short non-wake-word transcripts are rejected.

### Responses cut off mid-sentence

- Increase `llm_max_tokens` in config (default: 256)

### Conversation ends too quickly

- Increase `silence_timeout` (default: 15 seconds)
- Say "goodbye" or "stop" to end explicitly

### CUDA not detected for STT

Install [CUDA Toolkit 12](https://developer.nvidia.com/cuda-downloads) and ensure `stt_device: "cuda"` is set in config.

---

## 📁 Project Structure

```
jarvis/
├── main.py                  # CLI entry point (voice, cli, test modes)
├── config.yaml              # Root configuration
├── config/
│   └── settings.yaml        # Runtime settings
├── requirements.txt         # Python dependencies
├── api/                     # FastAPI backend
│   ├── main.py              #   App setup, CORS, routes
│   ├── websocket.py         #   WebSocket endpoint
│   └── routes/              #   API route handlers
│       ├── conversation.py  #     Chat (GET/POST/DELETE)
│       ├── system.py        #     CPU, RAM, disk stats
│       ├── weather.py       #     Open-Meteo weather
│       ├── uptime.py        #     Session uptime + commands
│       └── voice.py         #     Voice pipeline status
├── ui/                      # Next.js web dashboard
│   ├── src/
│   │   ├── app/             #     Pages + layout
│   │   ├── components/      #     React components
│   │   └── lib/             #     API client + WebSocket
│   └── package.json
├── agent/                   # LLM agent graph
│   ├── graph.py             #   Agent graph + session state
│   ├── nodes.py             #   Input, Reflector, Tool, Output nodes
│   ├── llm_factory.py       #   LLM provider factory
│   ├── tools.py             #   Tool definitions
│   ├── personality.py       #   JARVIS personality config
│   └── utils.py             #   Helpers (color_print, etc.)
├── voice/                   # Voice pipeline
│   ├── session.py           #   Voice session management
│   ├── stt.py               #   Speech-to-text (faster-whisper)
│   ├── tts.py               #   Text-to-speech (Piper)
│   ├── wake.py              #   Wake word listener (VAD + fuzzy match)
│   ├── validation.py        #   Transcript validation + noise filtering
│   └── recorder.py          #   Audio recording
├── memory/                  # Memory systems
│   ├── short_term.py        #   Conversation window
│   ├── long_term.py         #   ChromaDB vector store
│   ├── sqlite_store.py      #   SQLite persistence
│   └── vector_store.py      #   Vector search
├── tools/                   # Tool implementations
│   ├── computer_control.py  #   Mouse/keyboard automation
│   └── system_monitor.py    #   CPU, RAM, disk stats
├── skills/                  # Skill plugin system
├── tests/                   # Unit tests
├── data/                    # Runtime data (gitignored)
├── log/                     # Log files (gitignored)
└── docs/                    # Documentation
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## ❤️ Sponsor

If JARVIS helps you, consider sponsoring the project:

<a href="https://github.com/sponsors/adirangel">
  <img src="https://img.shields.io/badge/%E2%9D%A4%EF%B8%8F_Sponsor_on_GitHub-ea4aaa?style=for-the-badge&logo=github-sponsors" alt="Sponsor on GitHub" />
</a>

Your support helps keep JARVIS 100% local and open source.

---

## 📄 License

MIT
