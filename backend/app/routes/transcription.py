"""Speech-to-text API — REAL faster-whisper ASR for audio.

POST /api/ai/transcribe/audio transcribes uploaded audio with the
local Whisper model (tiny, int8, CPU). If the model cannot load it
answers HTTP 503 — it never invents a transcript.

POST /api/ai/transcribe accepts client-supplied text (e.g. browser Web
Speech API) and returns it labelled engine "client-text-passthrough"
with a note — it is NOT Whisper output.

Live captions stream over WebSocket /ws/captions (see app/main.py).
"""
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from app.deps import get_current_user
from app.models import User
from app.schemas import TranscribeRequest, TranscribeResponse
from app.services import speech_service

router = APIRouter(prefix="/api/ai", tags=["ai-speech"])

MAX_AUDIO_BYTES = 10 * 1024 * 1024  # 10MB


@router.post("/transcribe", response_model=TranscribeResponse)
def transcribe(payload: TranscribeRequest):
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="Provide 'text' (client transcription) "
                            "or upload audio to /api/ai/transcribe/audio")
    r = speech_service.transcribe_text(payload.text, payload.language or "en")
    return TranscribeResponse(**r)


@router.post("/transcribe/audio", response_model=TranscribeResponse)
async def transcribe_audio(request: Request, file: UploadFile = File(...),
                           language: str = "en",
                           user: User = Depends(get_current_user)):
    """Whisper transcription requires authentication (and is
    rate-limited, see app/security.py): previously anyone could burn
    Whisper compute anonymously."""
    # Reject files over 10MB: Content-Length when present, otherwise cap
    # the bytes actually read (10MB + 1) so a huge body cannot be buffered.
    content_length = request.headers.get("content-length")
    if content_length and content_length.isdigit() and int(content_length) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio file exceeds the 10MB limit")
    audio = await file.read(MAX_AUDIO_BYTES + 1)
    if len(audio) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio file exceeds the 10MB limit")
    if not audio:
        raise HTTPException(status_code=400, detail="Empty audio file")
    try:
        r = await speech_service.transcribe_audio_bytes_async(audio, language)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return TranscribeResponse(**r)
