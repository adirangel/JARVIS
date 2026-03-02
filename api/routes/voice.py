"""Voice status endpoint - listening/speaking state."""

from fastapi import APIRouter

router = APIRouter(prefix="/api", tags=["voice"])

# Voice runs in separate process (main.py --mode voice); API returns status
_voice_status = "listening"  # listening | speaking | idle


@router.get("/voice/status")
async def get_voice_status():
    """Return current voice state."""
    return {"status": _voice_status, "message": "Listening for wake word..."}
