"""Voice-agent API — the LiveKit "SignSpeak Voice" participant.

POST /{room_id}/voice/start  — join the room as signspeak-voice
POST /{room_id}/voice/speak  — speak text into the room (auto-starts)
POST /{room_id}/voice/stop   — leave the room
GET  /{room_id}/voice/status — agent state for the room

Only room members (creator or participant) may use these endpoints.
Speak is serialised through the agent's queue: overlapping requests
queue instead of overlapping audio. 409 when an agent is already
running for the room (start); 503 when LiveKit/Piper is unavailable.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app import config
from app.database import get_db
from app.deps import get_current_user
from app.models import User
from app.routes.rooms import _find, _is_member
from app.schemas import VoiceSpeakRequest, VoiceStatusOut
from app.services import livekit_service, voice_agent

router = APIRouter(prefix="/api/ai/rooms", tags=["ai-voice-agent"])


def _require_member(room_id: str, db: Session, user: User):
    room = _find(db, room_id)
    if not room.is_active:
        raise HTTPException(status_code=410, detail="Room has ended")
    if not _is_member(db, room, user.id):
        raise HTTPException(status_code=403, detail="Not a participant of this room")
    return room


@router.post("/{room_id}/voice/start", response_model=VoiceStatusOut)
def voice_start(room_id: str, db: Session = Depends(get_db),
                user: User = Depends(get_current_user)):
    room = _require_member(room_id, db, user)
    if not config.VOICE_AGENT_ENABLED:
        raise HTTPException(status_code=503, detail="Voice agent is disabled (VOICE_AGENT_ENABLED=false).")
    if not livekit_service.livekit_configured():
        raise HTTPException(status_code=503, detail="LiveKit is not configured; the voice agent cannot join a room.")
    try:
        status = voice_agent.start_agent(room.id, room.code)
    except RuntimeError as exc:
        msg = str(exc)
        if "already running" in msg:
            raise HTTPException(status_code=409, detail=msg)
        raise HTTPException(status_code=503, detail=msg)
    return VoiceStatusOut(**status)


@router.post("/{room_id}/voice/speak")
def voice_speak(room_id: str, payload: VoiceSpeakRequest,
                db: Session = Depends(get_db),
                user: User = Depends(get_current_user)):
    room = _require_member(room_id, db, user)
    if not config.VOICE_AGENT_ENABLED:
        raise HTTPException(status_code=503, detail="Voice agent is disabled (VOICE_AGENT_ENABLED=false).")
    if not livekit_service.livekit_configured():
        raise HTTPException(status_code=503, detail="LiveKit is not configured; the voice agent cannot join a room.")
    try:
        result = voice_agent.speak_in_room(room.id, room.code, payload.text)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    if not result.get("spoken"):
        raise HTTPException(status_code=503, detail=result.get("error") or "Voice agent could not speak the text.")
    return {"spoken": True, "duration_s": result.get("duration_s", 0.0),
            "room_id": room.id, "identity": voice_agent.AGENT_IDENTITY}


@router.post("/{room_id}/voice/stop", response_model=VoiceStatusOut)
def voice_stop(room_id: str, db: Session = Depends(get_db),
               user: User = Depends(get_current_user)):
    room = _require_member(room_id, db, user)
    voice_agent.stop_agent(room.id)
    return VoiceStatusOut(**voice_agent.agent_status(room.id))


@router.get("/{room_id}/voice/status", response_model=VoiceStatusOut)
def voice_status(room_id: str, db: Session = Depends(get_db),
                 user: User = Depends(get_current_user)):
    room = _require_member(room_id, db, user)
    return VoiceStatusOut(**voice_agent.agent_status(room.id))
