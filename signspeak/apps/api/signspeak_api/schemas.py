from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from .models import Action, Modality, Source


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# --- auth -----------------------------------------------------------------
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=10, description="10+ chars; this is a real product, not a demo login")
    display_name: str = Field(min_length=1, max_length=120)
    preferred_modality: Modality = Modality.SPEECH
    dominant_hand: str = Field(default="right", pattern="^(left|right)$")


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(ORMModel):
    id: uuid.UUID
    email: EmailStr
    display_name: str
    preferred_modality: Modality
    dominant_hand: str
    caption_size: str
    high_contrast: bool
    reduced_motion: bool
    tts_voice: str | None


class PreferencesUpdate(BaseModel):
    preferred_modality: Modality | None = None
    dominant_hand: str | None = Field(default=None, pattern="^(left|right)$")
    caption_size: str | None = Field(default=None, pattern="^(small|medium|large|xlarge)$")
    high_contrast: bool | None = None
    reduced_motion: bool | None = None
    tts_voice: str | None = None


# --- rooms ----------------------------------------------------------------
class CreateRoomRequest(BaseModel):
    display_name: str = Field(min_length=1, max_length=120)
    modality: Modality = Modality.SPEECH
    transcript_enabled: bool = False


class JoinRoomRequest(BaseModel):
    display_name: str = Field(min_length=1, max_length=120)
    modality: Modality = Modality.SPEECH
    consent_transcript: bool = False


class RoomTokenResponse(BaseModel):
    room_name: str
    call_id: uuid.UUID
    identity: str
    token: str
    livekit_url: str
    transcript_enabled: bool
    server_time_ms: float


class ParticipantOut(ORMModel):
    identity: str
    display_name: str
    modality: Modality
    consented_transcript: bool
    joined_at: datetime
    left_at: datetime | None


class CallOut(ORMModel):
    id: uuid.UUID
    room_name: str
    started_at: datetime
    ended_at: datetime | None
    transcript_enabled: bool
    model_version: str | None
    participants: list[ParticipantOut] = []


# --- transcript -----------------------------------------------------------
class TranscriptEntryIn(BaseModel):
    """What a client posts. `started_at_ms` is the CLIENT monotonic clock; the
    server translates it via the participant's ClockSync offset before
    ordering. Never trust it as a wall clock."""

    source: Source
    text: str = Field(max_length=2000)
    raw_label: str | None = None
    runner_up_label: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    energy_score: float | None = None
    action: Action | None = None
    started_at_ms: float
    ended_at_ms: float
    is_final: bool = True
    utterance_id: str | None = None


class TranscriptEntryOut(ORMModel):
    id: uuid.UUID
    participant_identity: str
    source: Source
    action: Action | None
    text: str
    raw_label: str | None
    runner_up_label: str | None
    confidence: float | None
    started_at_ms: int
    ended_at_ms: int
    is_final: bool
    created_at: datetime


class RecognitionEventIn(BaseModel):
    predicted_label: str
    confidence: float = Field(ge=0.0, le=1.0)
    energy_score: float | None = None
    action: Action
    corrected_label: str | None = None
    segment_frames: int
    inference_ms: float | None = None
    end_to_end_ms: float | None = None
    model_version: str | None = None


class ClockSyncRequest(BaseModel):
    t1: float  # client send time (client monotonic ms)


class ClockSyncResponse(BaseModel):
    t1: float
    t2: float  # server receive
    t3: float  # server send
