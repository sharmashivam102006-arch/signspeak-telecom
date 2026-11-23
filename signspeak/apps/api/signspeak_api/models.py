"""SQLAlchemy models.

Note what is *not* stored: no video, no audio, no landmarks. The sign
classifier runs entirely in the browser and publishes only the recognised word
over a LiveKit data channel, so the most sensitive data in the system never
reaches a server we operate. That is a design decision worth a paragraph in the
report, and it is also why the privacy claim in the pitch is defensible rather
than aspirational.

What *is* stored is still sensitive. A transcript of a Deaf user's medical or
banking call is a record of a private conversation. Hence: transcripts are
opt-in per call, consent is recorded per participant, and rows carry an
expires_at that a scheduled job enforces.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_uuid() -> uuid.UUID:
    return uuid.uuid4()


class Base(DeclarativeBase):
    pass


class Modality(str, enum.Enum):
    """How a participant prefers to send and receive.

    This is a communication preference, not a statement about the person. The
    schema deliberately has no "is_deaf" column: the system needs to know which
    channel to render, and nothing more. Storing disability status would be
    both unnecessary and, in most of the target sectors, regulated.
    """

    SIGN = "sign"
    SPEECH = "speech"
    TEXT = "text"


class Source(str, enum.Enum):
    SIGN = "sign"       # sign classifier output
    SPEECH = "speech"   # ASR output
    TEXT = "text"       # typed by the user
    SYSTEM = "system"   # "connection lost", "recognition paused"


class Action(str, enum.Enum):
    """Mirrors calibrate.decide(). Stored so the transcript preserves what the
    system actually did, not just what it guessed. An examiner asking "what
    happened when it wasn't sure?" gets an answer from the data."""

    SPOKEN = "spoken"
    CONFIRMED = "confirmed"     # user confirmed a low-confidence guess
    CORRECTED = "corrected"     # user overrode the guess - the gold mine for error analysis
    REJECTED = "rejected"


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=new_uuid)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(120))
    password_hash: Mapped[str] = mapped_column(String(255))

    preferred_modality: Mapped[Modality] = mapped_column(
        Enum(Modality, name="modality"), default=Modality.SPEECH
    )
    dominant_hand: Mapped[str] = mapped_column(String(8), default="right")
    caption_size: Mapped[str] = mapped_column(String(16), default="medium")
    high_contrast: Mapped[bool] = mapped_column(Boolean, default=False)
    reduced_motion: Mapped[bool] = mapped_column(Boolean, default=False)
    tts_voice: Mapped[str | None] = mapped_column(String(80), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    calls: Mapped[list["Call"]] = relationship(back_populates="created_by")


class Call(Base):
    __tablename__ = "calls"

    id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=new_uuid)
    room_name: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    created_by_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Opt-in, and only meaningful once every participant has consented.
    transcript_enabled: Mapped[bool] = mapped_column(Boolean, default=False)

    # Which model version produced the sign labels in this call. Without it,
    # error analysis across a semester of retraining is guesswork.
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)

    created_by: Mapped["User | None"] = relationship(back_populates="calls")
    entries: Mapped[list["TranscriptEntry"]] = relationship(
        back_populates="call", cascade="all, delete-orphan"
    )
    participants: Mapped[list["Participant"]] = relationship(
        back_populates="call", cascade="all, delete-orphan"
    )


class Participant(Base):
    __tablename__ = "participants"
    __table_args__ = (UniqueConstraint("call_id", "identity", name="uq_participant_identity"),)

    id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=new_uuid)
    call_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("calls.id", ondelete="CASCADE"))
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    identity: Mapped[str] = mapped_column(String(120))       # LiveKit participant identity
    display_name: Mapped[str] = mapped_column(String(120))
    modality: Mapped[Modality] = mapped_column(Enum(Modality, name="modality"))

    # Consent is per person per call, with a timestamp, because "the host
    # ticked a box" is not consent from the other participant.
    consented_transcript: Mapped[bool] = mapped_column(Boolean, default=False)
    consented_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    joined_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    left_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    call: Mapped["Call"] = relationship(back_populates="participants")


class TranscriptEntry(Base):
    __tablename__ = "transcript_entries"
    __table_args__ = (
        # Ordering is by client start time, not insertion order - see
        # services/ordering.py for why.
        Index("ix_transcript_call_started", "call_id", "started_at_ms"),
    )

    id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=new_uuid)
    call_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("calls.id", ondelete="CASCADE"))
    participant_identity: Mapped[str] = mapped_column(String(120))

    source: Mapped[Source] = mapped_column(Enum(Source, name="entry_source"))
    action: Mapped[Action | None] = mapped_column(Enum(Action, name="entry_action"), nullable=True)

    text: Mapped[str] = mapped_column(Text)
    raw_label: Mapped[str | None] = mapped_column(String(64), nullable=True)   # e.g. "HELP"
    runner_up_label: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Calibrated, not raw softmax. Null for speech and typed text, because a
    # number there would imply a comparability that does not exist.
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    energy_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Client-side monotonic clock, offset-corrected against the server. Two
    # timestamps because a 1.4s sign that starts before a 0.3s word should be
    # ordered before it.
    started_at_ms: Mapped[int] = mapped_column(Integer)
    ended_at_ms: Mapped[int] = mapped_column(Integer)
    is_final: Mapped[bool] = mapped_column(Boolean, default=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: utcnow() + timedelta(days=30)
    )

    call: Mapped["Call"] = relationship(back_populates="entries")


class RecognitionEvent(Base):
    """Every decision the sign engine made, including the rejections.

    Optional, off by default, and the single most useful thing you can collect
    for the evaluation chapter: it is a record of real-world behaviour under
    distribution shift, which no offline test set gives you. Stores labels and
    scores only - never landmarks, never frames.
    """

    __tablename__ = "recognition_events"

    id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=new_uuid)
    call_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("calls.id", ondelete="CASCADE"))
    participant_identity: Mapped[str] = mapped_column(String(120))

    predicted_label: Mapped[str] = mapped_column(String(64))
    confidence: Mapped[float] = mapped_column(Float)
    energy_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    action: Mapped[Action] = mapped_column(Enum(Action, name="entry_action"))

    # Set only when the user corrected it. These rows are the labelled
    # in-the-wild errors that make the next model better.
    corrected_label: Mapped[str | None] = mapped_column(String(64), nullable=True)

    segment_frames: Mapped[int] = mapped_column(Integer)
    inference_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    end_to_end_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
