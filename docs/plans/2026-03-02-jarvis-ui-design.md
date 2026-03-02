# JARVIS UI Design Document

**Date:** 2026-03-02  
**Status:** Implemented

## Overview

Web-based JARVIS dashboard matching the reference image layout. Two-tier architecture: FastAPI backend + Next.js frontend.

## Layout

- **Header:** J.A.R.V.I.S. branding, Online status, clock, weather summary, settings
- **Left Sidebar (280px):** System Stats, Weather, Camera, System Uptime cards
- **Center:** Circular audio visualizer, JARVIS name, listening status, voice controls (camera, mic, mail)
- **Right Panel (360px):** Conversation with chat bubbles, Clear, Extract, message input

## Tech Stack

- **Backend:** FastAPI, uvicorn, httpx
- **Frontend:** Next.js 14, React 18, Tailwind CSS, TypeScript

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| /api/health | GET | Backend status |
| /api/system | GET | CPU, RAM, disk |
| /api/weather | GET | Weather (Open-Meteo) |
| /api/uptime | GET | Session uptime, command count |
| /api/conversation | GET/POST/DELETE | Chat history, send message, clear |
| /api/voice/status | GET | Voice state |
| /ws | WebSocket | Real-time updates |

## Design Tokens

- Background: #0d1117, #161b22, #21262d
- Text: #c9d1d9, #8b949e
- Accent: #58a6ff (blue), #3fb950 (green)
- Borders: #30363d

## Running

```bash
# Terminal 1: Start API
cd d:\JARVIS
uvicorn api.main:app --reload --port 8000

# Terminal 2: Start UI
cd d:\JARVIS\ui
npm run dev
```

Open http://localhost:3000
