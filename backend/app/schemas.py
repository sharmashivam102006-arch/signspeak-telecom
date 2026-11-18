"""Pydantic v2 schemas."""
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field


# --- Auth ---
class UserCreate(BaseModel):
    email: EmailStr
    name: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=8, max_length=128)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: str
    email: str
    name: str
    is_guest: bool = False
    created_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class GuestCreate(BaseModel):
    name: str = Field(default="Guest", max_length=100)


# --- Rooms ---
class RoomCreate(BaseModel):
    name: str = Field(default="SignSpeak Call", max_length=120)


class RoomOut(BaseModel):
    id: str
    code: str
    name: str
    creator_id: Optional[str] = None
    created_at: Optional[datetime] = None
    is_active: bool = True
    invite_link: Optional[str] = None

    model_config = {"from_attributes": True}


class RoomJoin(BaseModel):
    code: Optional[str] = None
    name: str = Field(default="Guest", max_length=100)


class ParticipantOut(BaseModel):
    user_id: Optional[str] = None
    identity: str = Field(max_length=100)
    name: str = Field(max_length=100)
    joined_at: Optional[datetime] = None
    left_at: Optional[datetime] = None
    is_active: bool = True

    model_config = {"from_attributes": True}


class LeaveResponse(BaseModel):
    room_id: str
    user_id: Optional[str] = None
    identity: str
    left_at: Optional[datetime] = None
    is_active: bool = False


class EndResponse(BaseModel):
    room_id: str
    code: str
    is_active: bool = False
    livekit_room_deleted: bool = False


class SignalMessage(BaseModel):
    type: str  # offer | answer | ice-candidate | caption | sign | tts | transcript
    sender: Optional[str] = None
    payload: dict = {}


# --- Transcripts ---
class TranscriptCreate(BaseModel):
    speaker_name: str = ""
    direction: str = "speech-to-text"
    text: str = Field(min_length=1)
    confidence: Optional[int] = Field(default=None, ge=0, le=100)
    # Explicit per-request consent to store this transcript entry server-side.
    # Storage is also allowed when the user's stored store_transcripts
    # preference is True (see transcript_service.add_entry).
    consent: bool = False


class TranscriptOut(BaseModel):
    id: str
    room_id: str
    speaker_name: str = ""
    direction: str
    text: str
    confidence: Optional[int] = None
    created_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


# --- AI: sign ---
class SignPredictRequest(BaseModel):
    # Payload caps: a hand has 21 landmarks; the LSTM consumes 30 frames.
    landmarks: List[List[float]] = Field(default_factory=list, max_length=21)  # 21 x [x,y,z]
    sequence: Optional[List[List[List[float]]]] = Field(default=None, max_length=30)  # frames x 21 x [x,y,z]
    force: bool = False  # bypass duplicate suppression
    demo: bool = False  # explicitly opt into the labelled demo pose heuristics


class SignPredictResponse(BaseModel):
    prediction: Optional[str] = None
    confidence: float = 0.0
    mode: str  # demo-rule-based | trained-model | no-hand | model-not-trained
    requires_confirm: bool = True
    message: str = ""
    vocabulary: List[str] = []


class SignCollectRequest(BaseModel):
    label: str = Field(min_length=1, max_length=50)
    sequence: List[List[List[float]]] = Field(min_length=5, max_length=30)
    # Sign-language label the contributor confirms for this sample.
    # Default "generic": introductory hand signs, NOT certified Indian
    # Sign Language. Pass "ISL" only when the sign is confirmed ISL.
    # There is deliberately NO signer_id field — the server sets the
    # signer from the authenticated user; client input is never trusted.
    language: str = Field(default="generic", max_length=30)


class SignTrainRequest(BaseModel):
    epochs: int = Field(default=30, ge=1, le=200)
    hidden_dim: int = Field(default=128, ge=16, le=512)


class TrainJobOut(BaseModel):
    job_id: str
    status: str  # queued | running | done | failed
    progress: float = 0.0
    epochs_done: int = 0
    epochs_total: int = 0
    val_accuracy: Optional[float] = None
    message: str = ""
    # Evaluation report for the finished job. val_available is False
    # and per_class/confusion_matrix are None when no validation data
    # existed — metrics are never fabricated.
    split: Optional[str] = None  # by_signer | random | none
    val_available: bool = False
    per_class: Optional[dict] = None  # {label: {precision, recall, f1, support}}
    confusion_matrix: Optional[dict] = None  # {labels, matrix}
    train_signers: List[str] = []
    val_signers: List[str] = []


# --- AI: speech / TTS ---
class TranscribeRequest(BaseModel):
    text: Optional[str] = None
    language: Optional[str] = "en"
    audio_duration_seconds: Optional[float] = None
    audio_format: Optional[str] = None


class TranscribeResponse(BaseModel):
    text: str
    language: str = "en"
    engine: str
    is_final: bool = True
    note: str = ""
    confidence: float = 0.0
    segments: List[dict] = []


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    voice: Optional[str] = None
    language: Optional[str] = "en"


class TTSResponse(BaseModel):
    text: str
    audio_url: Optional[str] = None
    engine: str
    voice: Optional[str] = None
    note: str = ""
    duration_s: Optional[float] = None


# --- AI: voice agent ---
class VoiceSpeakRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)


class VoiceStatusOut(BaseModel):
    room_id: str
    identity: str = "signspeak-voice"
    running: bool = False
    queue_depth: int = 0
    error: Optional[str] = None


# --- Preferences ---
class PreferencesOut(BaseModel):
    role: str = "hearing"
    captions_enabled: bool = True
    tts_enabled: bool = True
    tts_muted: bool = False
    tts_volume: int = 80
    caption_language: str = "en"
    store_transcripts: bool = False

    model_config = {"from_attributes": True}


class PreferencesUpdate(BaseModel):
    role: Optional[str] = None
    captions_enabled: Optional[bool] = None
    tts_enabled: Optional[bool] = None
    tts_muted: Optional[bool] = None
    tts_volume: Optional[int] = Field(default=None, ge=0, le=100)
    caption_language: Optional[str] = None
    store_transcripts: Optional[bool] = None
