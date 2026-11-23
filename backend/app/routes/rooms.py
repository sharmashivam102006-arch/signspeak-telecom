"""Room routes: create, list, get, join, leave, participants, end,
LiveKit token, REST signal fallback.

Media runs on LiveKit: clients join with the token from
GET /{room_id}/token (room name = room code, identity = user id).
Participant membership is tracked in the RoomParticipant table (upserted
on join, marked inactive on leave/end, and mirrored by the LiveKit
webhook). The legacy /ws/call WebSocket in app/main.py is untouched and
now only relays captions/sign messages between peers.
"""
import json
import logging
import random
import string
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.deps import get_current_user
from app.models import Room, RoomParticipant, User
from app.schemas import (EndResponse, LeaveResponse, ParticipantOut,
                         RoomCreate, RoomJoin, RoomOut, SignalMessage)
from app.services import livekit_service
from app.services.state_store import get_state_store, signal_queue_key
from app.services.livekit_service import room_url

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/rooms", tags=["rooms"])

# REST signalling fallback: pending messages per room, backed by the
# state store (in-memory by default; Redis when REDIS_URL is set, so the
# queue survives restarts and is shared across workers). Keyed by the
# room's DB id so id- and code-based lookups share one queue.


def _code() -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


def _out(room: Room) -> RoomOut:
    data = RoomOut.model_validate(room)
    data.invite_link = room_url(room.code)
    return data


def _find(db: Session, room_id: str) -> Room:
    room = db.query(Room).filter(Room.id == room_id).first() or \
        db.query(Room).filter(Room.code == room_id.upper()).first()
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    return room


def _is_member(db: Session, room: Room, user_id: str) -> bool:
    """Creator or (active or past) participant may see the roster."""
    if room.creator_id and room.creator_id == user_id:
        return True
    return db.query(RoomParticipant).filter(
        RoomParticipant.room_id == room.id,
        RoomParticipant.user_id == user_id).first() is not None


def _upsert_participant(db: Session, room: Room, user: User) -> RoomParticipant:
    """Record (or reactivate) the caller's membership in the room."""
    row = db.query(RoomParticipant).filter(
        RoomParticipant.room_id == room.id,
        RoomParticipant.user_id == user.id).first()
    if row is None:
        row = RoomParticipant(room_id=room.id, user_id=user.id,
                              identity=user.id, name=user.name or "Guest")
        db.add(row)
    else:
        row.is_active = True
        row.left_at = None
        row.name = user.name or row.name
    row.joined_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(row)
    return row


@router.post("", response_model=RoomOut, status_code=201)
def create_room(payload: RoomCreate, db: Session = Depends(get_db),
                user: User = Depends(get_current_user)):
    code = _code()
    while db.query(Room).filter(Room.code == code).first():
        code = _code()
    room = Room(code=code, name=payload.name, creator_id=user.id)
    db.add(room); db.commit(); db.refresh(room)
    return _out(room)


@router.get("", response_model=list[RoomOut])
def list_rooms(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    rooms = db.query(Room).filter(Room.creator_id == user.id).order_by(Room.created_at.desc()).all()
    return [_out(r) for r in rooms]


@router.get("/{room_id}", response_model=RoomOut)
def get_room(room_id: str, db: Session = Depends(get_db),
             user: User = Depends(get_current_user)):
    """Room lookup is authenticated and visible only to the creator or
    a (current or past) participant. Previously this returned the room
    code + invite link to anyone, unauthenticated — an anonymous caller
    could enumerate rooms and then request a LiveKit token for them.
    Invitees become members via POST /{room_id}/join (code required),
    which is the entry point for non-members."""
    room = _find(db, room_id)
    if not _is_member(db, room, user.id):
        raise HTTPException(status_code=403, detail="Not a participant of this room")
    return _out(room)


@router.post("/{room_id}/join", response_model=RoomOut)
def join_room(room_id: str, payload: RoomJoin = RoomJoin(),
              db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    room = _find(db, room_id)
    if payload.code and payload.code.upper() != room.code:
        raise HTTPException(status_code=400, detail="Invalid room code")
    if not room.is_active:
        raise HTTPException(status_code=410, detail="Room has ended")
    _upsert_participant(db, room, user)
    return _out(room)


@router.post("/{room_id}/leave", response_model=LeaveResponse)
def leave_room(room_id: str, db: Session = Depends(get_db),
               user: User = Depends(get_current_user)):
    """Mark the caller's participant row inactive (left_at=now)."""
    room = _find(db, room_id)
    # Two /join requests at the same moment can leave more than one row
    # for the same user; mark them ALL as left, or the roster keeps
    # showing someone who has gone.
    rows = db.query(RoomParticipant).filter(
        RoomParticipant.room_id == room.id,
        RoomParticipant.user_id == user.id).all()
    now = datetime.now(timezone.utc)
    if not rows:
        # Never joined via /join (e.g. joined LiveKit directly): record
        # the leave so the roster stays accurate.
        row = RoomParticipant(room_id=room.id, user_id=user.id,
                              identity=user.id, name=user.name or "Guest",
                              joined_at=now, left_at=now, is_active=False)
        db.add(row)
    else:
        for r in rows:
            r.is_active = False
            r.left_at = now
        row = rows[0]
    db.commit()
    db.refresh(row)
    return LeaveResponse(room_id=room.id, user_id=row.user_id,
                         identity=row.identity, left_at=row.left_at,
                         is_active=False)


@router.get("/{room_id}/participants", response_model=list[ParticipantOut])
def list_participants(room_id: str, db: Session = Depends(get_db),
                     user: User = Depends(get_current_user)):
    room = _find(db, room_id)
    if not _is_member(db, room, user.id):
        raise HTTPException(status_code=403, detail="Not a participant of this room")
    rows = (db.query(RoomParticipant)
            .filter(RoomParticipant.room_id == room.id)
            .order_by(RoomParticipant.joined_at.asc()).all())
    # One entry per person: if duplicate rows exist for an identity, the
    # most recently joined one wins (keeps join order otherwise).
    latest: dict[str, RoomParticipant] = {}
    for r in rows:
        latest[r.identity] = r
    seen: set[str] = set()
    out = []
    for r in rows:
        if r.identity in seen:
            continue
        seen.add(r.identity)
        out.append(ParticipantOut.model_validate(latest[r.identity]))
    return out


@router.post("/{room_id}/end", response_model=EndResponse)
def end_room(room_id: str, db: Session = Depends(get_db),
             user: User = Depends(get_current_user)):
    """Creator only: end the call. Marks all participants inactive and
    deletes the LiveKit room (best-effort — a LiveKit outage never 500s
    this endpoint; the DB state is authoritative)."""
    room = _find(db, room_id)
    if room.creator_id != user.id:
        raise HTTPException(status_code=403, detail="Only the room creator can end the call")
    now = datetime.now(timezone.utc)
    room.is_active = False
    db.query(RoomParticipant).filter(
        RoomParticipant.room_id == room.id,
        RoomParticipant.is_active.is_(True)).update(
            {"is_active": False, "left_at": now}, synchronize_session=False)
    db.commit()
    # Stop the voice agent for this room (best-effort; never 500s the end).
    try:
        from app.services import voice_agent
        voice_agent.stop_agent(room.id)
    except Exception as exc:  # noqa: BLE001 - agent teardown must not fail /end
        logger.warning("Voice agent stop failed for room %s: %s", room.id, exc)
    deleted = livekit_service.delete_room(room.code)
    if not deleted:
        logger.warning("LiveKit room %s could not be deleted (server down "
                       "or unconfigured); DB state still marks the room ended",
                       room.code)
    return EndResponse(room_id=room.id, code=room.code, is_active=False,
                       livekit_room_deleted=deleted)


@router.get("/{room_id}/token")
def room_token(room_id: str, db: Session = Depends(get_db),
               user: User = Depends(get_current_user)):
    """Real LiveKit token: room name = room code, identity = user id.

    503 when LiveKit is unconfigured — the frontend must show that
    calling is unavailable instead of falling back to a fake token.
    The creator gets room_admin=True (can end the room in LiveKit).
    Membership is verified first: only the creator or a participant
    (POST /join) may receive a token. Previously any authenticated
    user — including a fresh guest who had never joined — got a valid
    token for any room, which was an anonymous-to-media hole.
    """
    room = _find(db, room_id)
    if not room.is_active:
        raise HTTPException(status_code=410, detail="Room has ended")
    if not _is_member(db, room, user.id):
        raise HTTPException(status_code=403, detail="Not a participant of this room")
    result = livekit_service.generate_token(
        room.code, identity=user.id, name=user.name,
        room_admin=(room.creator_id == user.id))
    if result["mode"] != livekit_service.MODE_LIVEKIT or not result["token"]:
        raise HTTPException(status_code=503, detail=result["note"])
    return {"room_id": room.id, "code": room.code, **result}


@router.post("/{room_id}/signal")
def post_signal(room_id: str, msg: SignalMessage, db: Session = Depends(get_db),
                user: User = Depends(get_current_user)):
    """REST fallback for signalling when WebSocket is unavailable.
    Messages are queued in the state store and fetched by the peer via GET.
    Authenticated AND membership-checked: only the creator or a participant
    may inject messages into a room. Capped at 100 messages per room
    (oldest dropped) so it cannot grow without bound."""
    room = _find(db, room_id)
    if not _is_member(db, room, user.id):
        raise HTTPException(status_code=403, detail="Not a participant of this room")
    count = get_state_store().append(
        signal_queue_key(room.id), json.dumps(msg.model_dump()), max_len=100)
    return {"queued": True, "count": count}


@router.get("/{room_id}/signal")
def get_signal(room_id: str, db: Session = Depends(get_db),
               user: User = Depends(get_current_user)):
    room = _find(db, room_id)
    if not _is_member(db, room, user.id):
        raise HTTPException(status_code=403, detail="Not a participant of this room")
    msgs = [json.loads(raw) for raw in
            get_state_store().drain(signal_queue_key(room.id))]
    return {"messages": msgs}
