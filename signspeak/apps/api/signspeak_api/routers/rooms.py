"""Rooms and LiveKit access tokens.

The backend does not touch media. It mints a short-lived, narrowly-scoped
LiveKit token and gets out of the way; audio and video flow peer-to-peer or
through the SFU, never through this process. That separation is the reason the
system can scale AI work and media work independently, and it is also why a
FastAPI instance falling over drops new joins but not calls in progress.
"""

from __future__ import annotations

import secrets
import time
import uuid
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from livekit import api as lk
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..config import settings
from ..db import get_session
from ..deps import current_user_optional
from ..models import Call, Modality, Participant, User, utcnow
from ..schemas import CallOut, CreateRoomRequest, JoinRoomRequest, RoomTokenResponse

router = APIRouter(prefix="/rooms", tags=["rooms"])

# Ambiguity-free alphabet: no O/0, I/1/l. Room codes get read aloud over the
# phone and typed by someone who cannot hear them being read aloud.
ROOM_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"


def generate_room_name() -> str:
    return "-".join(
        "".join(secrets.choice(ROOM_ALPHABET) for _ in range(4)) for _ in range(2)
    )


def mint_token(room: str, identity: str, display_name: str, ttl_hours: int = 4) -> str:
    """Least-privilege LiveKit grant.

    room_create and room_admin are deliberately absent: a participant token
    that can destroy the room or kick others is a bug waiting for a bored
    undergraduate. Data channel publishing is allowed because that is how
    recognised signs and captions travel.
    """
    return (
        lk.AccessToken(settings.livekit_api_key, settings.livekit_api_secret)
        .with_identity(identity)
        .with_name(display_name)
        .with_ttl(timedelta(hours=ttl_hours))
        .with_grants(
            lk.VideoGrants(
                room_join=True,
                room=room,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
                can_update_own_metadata=True,
            )
        )
        .to_jwt()
    )


@router.post("", response_model=RoomTokenResponse, status_code=status.HTTP_201_CREATED)
async def create_room(
    body: CreateRoomRequest,
    session: AsyncSession = Depends(get_session),
    user: User | None = Depends(current_user_optional),
) -> RoomTokenResponse:
    """Create a room. Anonymous is allowed on purpose.

    Forcing a Deaf user to create an account before they can take an urgent
    call is an accessibility failure dressed up as security. Accounts exist to
    persist preferences, not to gate the product.
    """
    room_name = generate_room_name()
    call = Call(
        room_name=room_name,
        created_by_id=user.id if user else None,
        transcript_enabled=body.transcript_enabled,
    )
    session.add(call)
    await session.flush()

    identity = f"{_slug(body.display_name)}-{uuid.uuid4().hex[:6]}"
    session.add(
        Participant(
            call_id=call.id,
            user_id=user.id if user else None,
            identity=identity,
            display_name=body.display_name,
            modality=body.modality,
            consented_transcript=body.transcript_enabled,
            consented_at=utcnow() if body.transcript_enabled else None,
        )
    )
    await session.commit()

    return RoomTokenResponse(
        room_name=room_name,
        call_id=call.id,
        identity=identity,
        token=mint_token(room_name, identity, body.display_name),
        livekit_url=settings.livekit_url,
        transcript_enabled=call.transcript_enabled,
        server_time_ms=time.monotonic() * 1000.0,
    )


@router.post("/{room_name}/join", response_model=RoomTokenResponse)
async def join_room(
    room_name: str,
    body: JoinRoomRequest,
    session: AsyncSession = Depends(get_session),
    user: User | None = Depends(current_user_optional),
) -> RoomTokenResponse:
    room_name = room_name.upper().strip()
    call = (
        await session.execute(select(Call).where(Call.room_name == room_name))
    ).scalar_one_or_none()

    if call is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such room")
    if call.ended_at is not None:
        raise HTTPException(status.HTTP_410_GONE, "That call has ended")

    identity = f"{_slug(body.display_name)}-{uuid.uuid4().hex[:6]}"
    session.add(
        Participant(
            call_id=call.id,
            user_id=user.id if user else None,
            identity=identity,
            display_name=body.display_name,
            modality=body.modality,
            consented_transcript=body.consent_transcript,
            consented_at=utcnow() if body.consent_transcript else None,
        )
    )

    # Transcript stays off until everyone in the room has said yes. The host
    # enabling it is a request, not permission.
    if call.transcript_enabled and not body.consent_transcript:
        call.transcript_enabled = False

    await session.commit()

    return RoomTokenResponse(
        room_name=room_name,
        call_id=call.id,
        identity=identity,
        token=mint_token(room_name, identity, body.display_name),
        livekit_url=settings.livekit_url,
        transcript_enabled=call.transcript_enabled,
        server_time_ms=time.monotonic() * 1000.0,
    )


@router.get("/{room_name}", response_model=CallOut)
async def get_room(room_name: str, session: AsyncSession = Depends(get_session)) -> Call:
    call = (
        await session.execute(
            select(Call)
            .options(selectinload(Call.participants))
            .where(Call.room_name == room_name.upper().strip())
        )
    ).scalar_one_or_none()
    if call is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such room")
    return call


@router.post("/{room_name}/end", status_code=status.HTTP_204_NO_CONTENT)
async def end_room(room_name: str, session: AsyncSession = Depends(get_session)) -> None:
    call = (
        await session.execute(select(Call).where(Call.room_name == room_name.upper().strip()))
    ).scalar_one_or_none()
    if call is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such room")
    if call.ended_at is None:
        call.ended_at = utcnow()
    await session.commit()


def _slug(name: str) -> str:
    keep = [c.lower() for c in name if c.isalnum()]
    return ("".join(keep)[:20]) or "guest"
