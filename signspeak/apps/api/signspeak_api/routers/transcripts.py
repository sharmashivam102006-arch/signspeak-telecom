"""Transcript and recognition-event ingest.

Two endpoints that look similar and serve very different purposes:

  POST /calls/{id}/transcript   what the humans said, kept only with consent
  POST /calls/{id}/events       what the model decided, kept for evaluation

The second one is where the research value is. Every time a user overrides the
system's guess, that is a labelled error from the real distribution - the kind
of data no amount of careful dataset collection produces, because you cannot
anticipate what a stranger's living room looks like at 9pm.
"""

from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..db import get_session
from ..models import Action, Call, Participant, RecognitionEvent, Source, TranscriptEntry
from ..schemas import (
    ClockSyncRequest,
    ClockSyncResponse,
    RecognitionEventIn,
    TranscriptEntryIn,
    TranscriptEntryOut,
)
from ..services.ordering import merge_partials
from ..state import clock_for, sessions

router = APIRouter(prefix="/calls", tags=["transcript"])


@router.post("/clock-sync", response_model=ClockSyncResponse)
async def clock_sync(body: ClockSyncRequest, identity: str = Query(...)) -> ClockSyncResponse:
    """NTP-style round trip so the server can order two clients' events.

    Call it three or four times on join; the client keeps the sample with the
    smallest round trip. Without this, ordering by client timestamps orders by
    whose machine booted first. See services/ordering.py.
    """
    t2 = time.monotonic() * 1000.0
    sync = clock_for(identity)
    t3 = time.monotonic() * 1000.0
    sync.samples += 1
    return ClockSyncResponse(t1=body.t1, t2=t2, t3=t3)


async def _load_call(session: AsyncSession, call_id: uuid.UUID) -> Call:
    call = (await session.execute(select(Call).where(Call.id == call_id))).scalar_one_or_none()
    if call is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such call")
    return call


@router.post(
    "/{call_id}/transcript",
    response_model=TranscriptEntryOut | None,
    status_code=status.HTTP_202_ACCEPTED,
)
async def add_entry(
    call_id: uuid.UUID,
    body: TranscriptEntryIn,
    identity: str = Query(..., description="LiveKit participant identity"),
    session: AsyncSession = Depends(get_session),
):
    """Ingest one transcript entry.

    Partials are broadcast to the room and never persisted - they exist for the
    live caption rail, which needs immediacy, not durability. Only finals reach
    Postgres, and only if the call has consent.
    """
    call = await _load_call(session, call_id)

    participant = (
        await session.execute(
            select(Participant).where(
                Participant.call_id == call_id, Participant.identity == identity
            )
        )
    ).scalar_one_or_none()
    if participant is None:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Not a participant in this call")

    sync = clock_for(identity)
    started = int(sync.to_server(body.started_at_ms))
    ended = int(sync.to_server(body.ended_at_ms))

    # Always fan out live, ordered or not - captions must not wait on the
    # reorder window.
    await sessions.broadcast(
        call.room_name,
        {
            "type": "caption",
            "identity": identity,
            "display_name": participant.display_name,
            "source": body.source.value,
            "text": body.text,
            "confidence": body.confidence,
            "action": body.action.value if body.action else None,
            "is_final": body.is_final,
            "started_at_ms": started,
            "utterance_id": body.utterance_id,
        },
    )

    if not body.is_final:
        return None
    if not (call.transcript_enabled and participant.consented_transcript):
        return None
    if not settings.store_transcripts_by_default and not call.transcript_enabled:
        return None

    entry = TranscriptEntry(
        call_id=call_id,
        participant_identity=identity,
        source=body.source,
        action=body.action,
        text=body.text,
        raw_label=body.raw_label,
        runner_up_label=body.runner_up_label,
        confidence=body.confidence,
        energy_score=body.energy_score,
        started_at_ms=started,
        ended_at_ms=ended,
        is_final=True,
    )
    session.add(entry)
    await session.commit()
    await session.refresh(entry)
    return entry


@router.get("/{call_id}/transcript", response_model=list[TranscriptEntryOut])
async def get_transcript(
    call_id: uuid.UUID, session: AsyncSession = Depends(get_session)
) -> list[TranscriptEntry]:
    """Ordered by utterance START, not insertion.

    A 1.4s sign that began before a 0.3s spoken word belongs above it, even
    though it arrived second. Getting this wrong is what makes a demo
    transcript read as if the two people were talking past each other.
    """
    rows = (
        await session.execute(
            select(TranscriptEntry)
            .where(TranscriptEntry.call_id == call_id)
            .order_by(TranscriptEntry.started_at_ms, TranscriptEntry.created_at)
        )
    ).scalars().all()
    return list(rows)


@router.post("/{call_id}/events", status_code=status.HTTP_204_NO_CONTENT)
async def add_recognition_event(
    call_id: uuid.UUID,
    body: RecognitionEventIn,
    identity: str = Query(...),
    session: AsyncSession = Depends(get_session),
) -> None:
    """Log a model decision, including rejections and user corrections.

    Labels and scores only. No landmarks, no frames - the raw signal that could
    identify the signer never leaves their machine, which is the whole basis of
    the privacy claim.
    """
    await _load_call(session, call_id)
    session.add(
        RecognitionEvent(
            call_id=call_id,
            participant_identity=identity,
            predicted_label=body.predicted_label,
            confidence=body.confidence,
            energy_score=body.energy_score,
            action=body.action,
            corrected_label=body.corrected_label,
            segment_frames=body.segment_frames,
            inference_ms=body.inference_ms,
            end_to_end_ms=body.end_to_end_ms,
            model_version=body.model_version,
        )
    )
    await session.commit()


@router.get("/{call_id}/events/summary")
async def events_summary(call_id: uuid.UUID, session: AsyncSession = Depends(get_session)) -> dict:
    """In-the-wild behaviour for one call. Feed this into the report.

    `correction_rate` is the number that matters most and is absent from the
    original evaluation plan: how often did a human have to fix the system?
    Offline accuracy does not measure that, because offline there is no human.
    """
    rows = (
        await session.execute(
            select(RecognitionEvent).where(RecognitionEvent.call_id == call_id)
        )
    ).scalars().all()
    if not rows:
        return {"n": 0}

    spoken = [r for r in rows if r.action == Action.SPOKEN]
    corrected = [r for r in rows if r.corrected_label]
    lat = [r.end_to_end_ms for r in rows if r.end_to_end_ms is not None]
    lat.sort()

    return {
        "n": len(rows),
        "spoken": len(spoken),
        "confirmed": sum(1 for r in rows if r.action == Action.CONFIRMED),
        "rejected": sum(1 for r in rows if r.action == Action.REJECTED),
        "corrections": len(corrected),
        "correction_rate": len(corrected) / len(spoken) if spoken else None,
        "mean_confidence_spoken": (
            sum(r.confidence for r in spoken) / len(spoken) if spoken else None
        ),
        "end_to_end_ms_p50": lat[len(lat) // 2] if lat else None,
        "end_to_end_ms_p95": lat[int(len(lat) * 0.95)] if lat else None,
        "confusions": [
            {"predicted": r.predicted_label, "actual": r.corrected_label} for r in corrected
        ],
    }


@router.delete("/{call_id}/transcript", status_code=status.HTTP_204_NO_CONTENT)
async def delete_transcript(
    call_id: uuid.UUID, session: AsyncSession = Depends(get_session)
) -> None:
    """Either participant can erase the record of the conversation, at any
    time, without explanation. Not a feature request - the baseline for a
    product that sits inside private medical and financial conversations."""
    rows = (
        await session.execute(
            select(TranscriptEntry).where(TranscriptEntry.call_id == call_id)
        )
    ).scalars().all()
    for r in rows:
        await session.delete(r)
    await session.commit()
