"""LiveKit webhook receiver: POST /api/livekit/webhook.

LiveKit posts events (participant_joined, participant_left,
room_finished, ...) to this URL when configured in the LiveKit server
or Cloud dashboard. The DB roster (RoomParticipant) is kept in sync:

* participant_joined  -> upsert an active participant row
* participant_left    -> mark the row inactive (left_at=now)
* room_finished       -> mark the room inactive + all participants inactive

Signature verification uses the SDK's WebhookReceiver when LiveKit is
configured; a bad signature returns 200 {received: false} (no retry
storm — LiveKit retries 5xx responses). When LiveKit is NOT
configured the webhook FAILS CLOSED with HTTP 503: unverified events
are never applied to the database (previously they were accepted with
only a warning, so a forged `room_finished` could kill any room).
A malformed event always returns 200 {received: false} after logging.
"""
import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Room, RoomParticipant
from app.services import livekit_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/livekit", tags=["livekit-webhook"])


def _find_room_by_code(db: Session, code: str):
    if not code:
        return None
    return db.query(Room).filter(Room.code == code.upper()).first()


@router.post("/webhook")
async def livekit_webhook(request: Request, db: Session = Depends(get_db)):
    raw = await request.body()
    body = raw.decode("utf-8", errors="replace")
    auth_token = request.headers.get("Authorization", "")

    verified = False
    if livekit_service.livekit_configured():
        try:
            livekit_service.verify_webhook(body, auth_token)
            verified = True
        except Exception as exc:
            logger.warning("LiveKit webhook signature verification failed: %s", exc)
            return {"received": False, "verified": False,
                    "error": "webhook signature verification failed"}
    else:
        # FAIL CLOSED: without LiveKit credentials the HMAC signature
        # cannot be verified, so no event may mutate the database.
        # (Previously this branch accepted the event with only a
        # warning — a forged room_finished could end any room.)
        logger.warning("LiveKit webhook received but LiveKit is not "
                       "configured — rejecting unverified event")
        raise HTTPException(
            status_code=503,
            detail="LiveKit is not configured; webhook signature cannot "
                   "be verified")

    try:
        data = json.loads(body) if body.strip() else {}
    except json.JSONDecodeError:
        logger.warning("LiveKit webhook: malformed JSON body")
        return {"received": False, "verified": verified, "error": "malformed JSON"}

    event = data.get("event", "")
    room_data = data.get("room") or {}
    room = _find_room_by_code(db, room_data.get("name", ""))
    if room is None:
        logger.info("LiveKit webhook %s for unknown room %r — ignored",
                    event, room_data.get("name"))
        return {"received": True, "verified": verified, "event": event,
                "note": "unknown room"}

    now = datetime.now(timezone.utc)
    participant_data = data.get("participant") or {}
    identity = participant_data.get("identity", "")

    if event == "participant_joined" and identity:
        row = db.query(RoomParticipant).filter(
            RoomParticipant.room_id == room.id,
            RoomParticipant.identity == identity).first()
        if row is None:
            row = RoomParticipant(room_id=room.id, user_id=None,
                                  identity=identity,
                                  name=participant_data.get("name") or identity,
                                  joined_at=now, is_active=True)
            db.add(row)
        else:
            row.is_active = True
            row.left_at = None
            row.joined_at = now
            row.name = participant_data.get("name") or row.name
        db.commit()
    elif event == "participant_left" and identity:
        # All rows for this identity (duplicates can exist), not just one.
        db.query(RoomParticipant).filter(
            RoomParticipant.room_id == room.id,
            RoomParticipant.identity == identity).update(
                {"is_active": False, "left_at": now}, synchronize_session=False)
        db.commit()
    elif event == "room_finished":
        room.is_active = False
        db.query(RoomParticipant).filter(
            RoomParticipant.room_id == room.id,
            RoomParticipant.is_active.is_(True)).update(
                {"is_active": False, "left_at": now}, synchronize_session=False)
        db.commit()
    else:
        logger.info("LiveKit webhook: unhandled event %r for room %s", event, room.code)

    return {"received": True, "verified": verified, "event": event}
