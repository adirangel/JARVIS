"""WebSocket endpoint for real-time updates."""

import asyncio
import json
from fastapi import WebSocket, WebSocketDisconnect

# Connected clients
_clients: set[WebSocket] = set()


async def broadcast(message: dict):
    """Send message to all connected clients."""
    data = json.dumps(message)
    dead = set()
    for ws in _clients:
        try:
            await ws.send_text(data)
        except Exception:
            dead.add(ws)
    for ws in dead:
        _clients.discard(ws)


async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connection."""
    await websocket.accept()
    _clients.add(websocket)
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
