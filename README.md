# JARVIS v0.1

**100% local AI assistant** with voice (wake word, STT, TTS), Hebrew-first, and Paul Bettany's JARVIS personality.

---

## Requirements

| Item | Notes |
|------|-------|
| **OS** | Windows 10/11 |
| **Python** | 3.10+ |
| **Ollama** | Running locally |
| **GPU** (optional) | NVIDIA with CUDA 12 for faster STT |
| **Microphone** | For voice mode |

---

## Installation

### 1. Clone and install dependencies

```powershell
git clone <repo-url> jarvis
cd jarvis
git checkout dev
```

### 2. Create virtual environment (recommended)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Python packages

```powershell
pip install -r requirements.txt
```

### 4. Install Ollama and pull models

1. Install [Ollama](https://ollama.com)
2. Pull the required models:

```powershell
ollama pull aminadaven/dictalm2.0-instruct:q5_K_M
ollama pull qwen3:4b
ollama pull nomic-embed-text
```

### 5. Configure (optional)

JARVIS uses `config.yaml` if present, otherwise `config.example.yaml`. To customize:

```powershell
copy config.example.yaml config.yaml
```

Edit `config.yaml` to set:
- `wake_word.device` – microphone ID (run `python test_mic.py` to find it)
- `voice.stt_device` – `"cuda"` or `"cpu"`
- `tools.allowed_directories` – paths JARVIS can access

---

## Running JARVIS

### Voice mode (wake word)

Say **"Hey Jarvis"** then speak. After the first wake, JARVIS listens continuously (no need to repeat "Hey Jarvis") until you're silent for ~15 seconds or say "goodbye"/"stop".

```powershell
python main.py --mode voice
```

### Push-to-talk

Press Enter, speak, then stay silent to finish.

```powershell
python main.py --mode voice --push-to-talk
```

### Console mode (text only)

Type messages instead of speaking.

```powershell
python main.py --console
```

### System tray (default)

Runs in the background. Right-click the tray icon → **Start**.

```powershell
python main.py
```

---

## Verify setup

Test the voice pipeline (mic → STT → agent → TTS):

```powershell
python verify_voice.py
```

Test microphone levels:

```powershell
python test_mic.py
```

---

## CUDA (optional, for faster STT)

For GPU-accelerated speech-to-text, install [CUDA 12](https://developer.nvidia.com/cuda-downloads). See [docs/CUDA_SETUP.md](docs/CUDA_SETUP.md).

---

## Latency optimization (sub-2s on RTX 4080)

- **FastPath**: Simple commands (hi, thanks, שלום) skip Planner → direct Reflector
- **Streaming TTS**: First phrase ("As you wish, Sir...") plays within ~800ms
- **Hybrid LLM**: DictaLM for Planner/Reflector, Qwen3 for tools
- **Voice**: faster-whisper int8 + beam_size=3, Piper preload
- **Memory**: Chroma cache for <100ms queries
- **Debug timing**: Set `timing: true` in config for per-node latency

See [docs/OLLAMA_LATENCY.md](docs/OLLAMA_LATENCY.md) for Ollama flags (`--flash-attn`, `num_ctx`).

---

## Troubleshooting

### False positives (JARVIS responds to background noise)

If JARVIS triggers on silence or invents words like "thank you" when no one is speaking:

1. **Increase wake confidence** – In `config.yaml`, set `wake_word.wake_confidence` to `0.8` or `0.85` (higher = stricter).
2. **Noise gate** – Ensure `wake_word.noise_gate_rms` is set (e.g. `0.005`) to skip very quiet chunks.
3. **Min audio length** – Set `voice.min_audio_length` to `1.5` to ignore short noise bursts.
4. **VAD** – Keep `voice.use_vad: true` so STT filters silence/background.
5. **Debug** – Run with `--mode voice` or set `debug: true` to see rejected transcriptions in the log.

### Responses cut off mid-sentence

- JARVIS appends "Pardon the interruption, Sir." when the response is truncated.
- Increase `llm.max_tokens` in config if responses are often cut short.
- Streaming TTS buffers until sentence boundaries; check `voice.stream_tts: true`.

### Conversation ends too quickly

- Increase `voice.silence_timeout` (default 15s) to wait longer before ending the session.
- Say "goodbye" or "stop" to end explicitly.

---

## What is excluded from git (.gitignore)

- `data/` – databases and conversation history
- `_backup/` – local backups
- `config.local.yaml` – local overrides (if you add this)
- `.env` – secrets (never commit)

---

## Project structure

```
jarvis/
├── main.py              # Entry point
├── config.yaml          # Configuration (create from config.example.yaml)
├── config.example.yaml  # Template config
├── requirements.txt
├── agent/               # LangGraph, LLM, tools
├── voice/               # STT, TTS, wake word
├── memory/              # SQLite + ChromaDB
├── heartbeat.py         # Periodic task reminders
├── verify_voice.py      # Voice pipeline test
├── test_mic.py          # Microphone test
└── docs/
```

---

## License

MIT
