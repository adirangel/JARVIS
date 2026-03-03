"""Voice status endpoint - real-time listening/speaking state via WS bridge."""

from fastapi import APIRouter

from api.websocket import get_voice_status, set_voice_status

router = APIRouter(prefix="/api", tags=["voice"])


@router.get("/voice/status")
async def voice_status():
    """Return current voice state (bridged from voice process via WS module)."""
    status = get_voice_status()
    labels = {
        "listening": "Listening for wake word...",
        "speaking": "Speaking...",
        "processing": "Processing...",
        "idle": "Awaiting activation...",
    }
    return {"status": status, "message": labels.get(status, status)}


@router.post("/voice/status")
async def update_voice_status(body: dict):
    """Called by voice process to push status changes (listening/speaking/idle)."""
    new_status = body.get("status", "idle")
    set_voice_status(new_status)
    return {"ok": True}
