"""Transcript routes (consent-gated storage).

Access restriction: only the room's creator may list, add or export
transcripts for now. There is no room-participant model yet, so there is
no reliable way to know who else was in the call — restricting to the
creator is the safe default until participants are modelled. Revisit
when a participant/membership model exists.
"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.deps import get_current_user
from app.models import Room, User
from app.schemas import TranscriptCreate, TranscriptOut
from app.services import transcript_service

router = APIRouter(prefix="/api/rooms", tags=["transcripts"])


def _room(db, room_id):
    room = db.query(Room).filter(Room.id == room_id).first() or \
        db.query(Room).filter(Room.code == room_id.upper()).first()
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    return room


def _require_creator(room: Room, user: User):
    # No participant model exists yet — creator-only access (see module docstring).
    if room.creator_id != user.id:
        raise HTTPException(
            status_code=403,
            detail="Only the room creator can access this room's transcripts.",
        )


@router.get("/{room_id}/transcripts", response_model=list[TranscriptOut])
def list_transcripts(room_id: str, db: Session = Depends(get_db),
                     user: User = Depends(get_current_user)):
    room = _room(db, room_id)
    _require_creator(room, user)
    return transcript_service.list_entries(db, room.id)


@router.post("/{room_id}/transcripts", response_model=TranscriptOut, status_code=201)
def add_transcript(room_id: str, payload: TranscriptCreate, db: Session = Depends(get_db),
                   user: User = Depends(get_current_user)):
    room = _room(db, room_id)
    _require_creator(room, user)
    try:
        entry = transcript_service.add_entry(
            db, room.id, payload.direction, payload.text,
            payload.speaker_name, user.id, payload.confidence,
            consent=payload.consent)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return entry


@router.get("/{room_id}/transcripts/export", response_class=PlainTextResponse)
def export_transcripts(room_id: str, db: Session = Depends(get_db),
                       user: User = Depends(get_current_user)):
    room = _room(db, room_id)
    _require_creator(room, user)
    entries = transcript_service.list_entries(db, room.id)
    return transcript_service.export_txt(entries)
