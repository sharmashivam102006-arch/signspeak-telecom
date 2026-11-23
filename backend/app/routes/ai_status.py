"""GET /api/ai/status — one honest summary of every AI pipeline."""
from fastapi import APIRouter

from app import config
from app.services import (sign_data, sign_model_service, speech_service,
                          tts_service, vision_service, voice_agent)

router = APIRouter(prefix="/api/ai", tags=["ai-status"])


@router.get("/status")
def ai_status():
    counts = sign_data.counts()
    return {
        "sign": {
            "mode": ("trained-model" if sign_model_service.trained_available()
                     else "model-not-trained"),
            "trained": sign_model_service.trained_available(),
            "torch_available": sign_model_service.torch_available(),
            "mediapipe_available": vision_service.mediapipe_available(),
            "samples_per_sign": counts["samples_per_sign"],
        },
        "whisper": {
            "available": speech_service.whisper_available(),
            "model": config.WHISPER_MODEL,
            "engine": speech_service.ENGINE_NAME,
        },
        "tts": {
            "available": tts_service.piper_available() and tts_service.voices_available(),
            "engine": "piper",
            "voices": [v["id"] for v in tts_service.list_voices()],
        },
        "voice_agent": {
            "enabled": config.VOICE_AGENT_ENABLED,
            "rooms_active": voice_agent.rooms_active(),
        },
    }
