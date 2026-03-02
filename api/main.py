"""JARVIS API - FastAPI backend for the dashboard UI."""

import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.websockets import WebSocket

from api.routes import system, weather, uptime, conversation, voice
from api.websocket import websocket_endpoint

app = FastAPI(
    title="JARVIS API",
    description="Backend API for JARVIS dashboard",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(system.router)
app.include_router(weather.router)
app.include_router(uptime.router)
app.include_router(conversation.router)
app.include_router(voice.router)


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket_endpoint(websocket)


@app.get("/api/health")
async def health():
    """Backend online status."""
    return {"status": "ok", "service": "jarvis"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
