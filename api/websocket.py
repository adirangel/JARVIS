"""WebSocket endpoint for real-time updates (conversation, streaming, voice status)."""

import asyncio
import json
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger

# Connected clients
_clients: set[WebSocket] = set()

# Shared voice status (updated by voice bridge)
_voice_status: str = "idle"


def set_voice_status(status: str) -> None:
    """Update voice status and schedule a broadcast (call from any thread)."""
    global _voice_status
    _voice_status = status
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(broadcast({"type": "status", "status": status}))
    except RuntimeError:
        pass


def get_voice_status() -> str:
    return _voice_status


async def broadcast(message: dict) -> None:
    """Send message to all connected WebSocket clients."""
    if not _clients:
        return
    data = json.dumps(message)
    dead: set[WebSocket] = set()
    for ws in _clients:
        try:
            await ws.send_text(data)
        except Exception:
            dead.add(ws)
    _clients -= dead


async def broadcast_conversation(messages: list[dict]) -> None:
    """Push full conversation history to all clients."""
    await broadcast({"type": "conversation", "messages": messages})


async def broadcast_token(token: str, done: bool = False) -> None:
    """Push a single streaming LLM token to all clients."""
    await broadcast({"type": "stream", "token": token, "done": done})


async def websocket_endpoint(websocket: WebSocket) -> None:
    """Handle WebSocket connection with ping/pong keepalive."""
    await websocket.accept()
    _clients.add(websocket)
    logger.debug(f"[WS] Client connected ({len(_clients)} total)")
    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        pass
    finally:
        _clients.discard(websocket)
        logger.debug(f"[WS] Client disconnected ({len(_clients)} remaining)")
