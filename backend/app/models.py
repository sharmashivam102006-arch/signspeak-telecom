"""Database models: User, Room, TranscriptEntry, UserPreferences."""
import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from app.database import Base


def _utcnow():
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False, default="")
    hashed_password = Column(String, nullable=False, default="")
    is_guest = Column(Boolean, default=False)
    created_at = Column(DateTime, default=_utcnow)
    rooms = relationship("Room", back_populates="creator")
    preferences = relationship("UserPreferences", back_populates="user", uselist=False)


class Room(Base):
    __tablename__ = "rooms"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    code = Column(String, unique=True, index=True, nullable=False)  # 6-char join code
    name = Column(String, nullable=False, default="SignSpeak Call")
    creator_id = Column(String, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=_utcnow)
    is_active = Column(Boolean, default=True)
    creator = relationship("User", back_populates="rooms")
    transcripts = relationship("TranscriptEntry", back_populates="room", cascade="all, delete-orphan")
    participants = relationship("RoomParticipant", back_populates="room", cascade="all, delete-orphan")


class RoomParticipant(Base):
    """One user's membership in a room (one row per user+room).

    `identity` is the LiveKit identity (= our user id) and `name` the
    display name shown in the call. A row is reactivated (is_active=True,
    left_at=None) when the user rejoins; `leave` / room end mark it
    inactive. LiveKit webhook events (participant_joined/left) upsert
    rows here too, so the DB mirrors the real room state.
    """
    __tablename__ = "room_participants"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    room_id = Column(String, ForeignKey("rooms.id"), nullable=False, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=True, index=True)
    identity = Column(String, nullable=False, default="")
    name = Column(String, nullable=False, default="")
    joined_at = Column(DateTime, default=_utcnow)
    left_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    room = relationship("Room", back_populates="participants")


class TranscriptEntry(Base):
    __tablename__ = "transcript_entries"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    room_id = Column(String, ForeignKey("rooms.id"), nullable=False, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    speaker_name = Column(String, default="")
    # direction: "sign-to-speech" | "speech-to-text" | "system"
    direction = Column(String, nullable=False, default="speech-to-text")
    text = Column(Text, nullable=False)
    confidence = Column(Integer, nullable=True)  # 0-100, null when n/a
    created_at = Column(DateTime, default=_utcnow)
    room = relationship("Room", back_populates="transcripts")


class UserPreferences(Base):
    __tablename__ = "user_preferences"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), unique=True, nullable=False)
    role = Column(String, default="hearing")  # deaf | hearing
    captions_enabled = Column(Boolean, default=True)
    tts_enabled = Column(Boolean, default=True)
    tts_muted = Column(Boolean, default=False)
    tts_volume = Column(Integer, default=80)
    caption_language = Column(String, default="en")
    sign_threshold_high = Column(Integer, default=85)
    sign_threshold_low = Column(Integer, default=60)
    store_transcripts = Column(Boolean, default=False)  # explicit consent only
    user = relationship("User", back_populates="preferences")
