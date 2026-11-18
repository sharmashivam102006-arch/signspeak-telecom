"""Text-to-speech API — REAL Piper neural TTS.

POST /api/ai/tts synthesises a WAV with Piper and returns its
/api/ai/tts/audio/<id> URL. 503 when Piper or the voice is unavailable.
Text is capped at 500 characters (config.TTS_MAX_CHARS).
Synthesis requires authentication (and is rate-limited, see
app/security.py): previously anyone could burn Piper compute
anonymously.
"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from app.deps import get_current_user
from app.models import User
from app.schemas import TTSRequest, TTSResponse
from app.services import tts_service

router = APIRouter(prefix="/api/ai", tags=["ai-tts"])


@router.post("/tts", response_model=TTSResponse)
async def tts(payload: TTSRequest, user: User = Depends(get_current_user)):
    try:
        r = await tts_service.synthesize_async(
            payload.text, payload.voice or None, payload.language or "en")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return TTSResponse(**r)


@router.get("/tts/voices")
def voices():
    """Piper voices actually installed on disk."""
    return {"voices": tts_service.list_voices(),
            "engine": "piper",
            "available": tts_service.voices_available()}


@router.get("/tts/audio/{audio_id}")
def tts_audio(audio_id: str):
    """Serve a synthesised WAV file by id."""
    path = tts_service.audio_path(audio_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(path, media_type="audio/wav",
                        filename=f"signspeak-{audio_id}.wav")
