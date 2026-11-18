"""SignSpeak Telecom — FastAPI backend.

WebSocket signalling: /ws/call/{room_id}?token=<jwt>
Broadcasts JSON {type, sender, payload} to other peers in the room.
Message types: offer, answer, ice-candidate, join, leave, caption,
sign, tts, transcript. In-memory room registry; disconnect cleans up.
AI failures never terminate a call — signalling is independent of AI.
"""
import asyncio
import io
import json
import logging
from collections import defaultdict

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from app import config
from app.database import Base, SessionLocal, engine
from app.deps import decode_token
from app.models import Room
from app.routes import (ai_status, auth, livekit_webhook, rooms,
                        sign_recognition, transcription, transcripts, tts,
                        users, voice_agent)
from app.security import RateLimitMiddleware, limiter
from app.services import livekit_service, speech_service

logger = logging.getLogger(__name__)

_WS_ALLOWED_TYPES = {
    "join", "leave", "offer", "answer", "ice-candidate", "ice",
    "caption", "sign", "tts", "transcript",
}
_WS_MAX_BYTES = 64 * 1024  # 64KB per message

Base.metadata.create_all(bind=engine)

app = FastAPI(title="SignSpeak Telecom API", version="0.1.0",
              description="AI-powered bidirectional sign-language / speech communication platform.")
app.add_middleware(
    CORSMiddleware,
    # Explicit allowlist only — never "*" with allow_credentials=True
    # (browsers reject that combination and it is insecure).
    allow_origins=config.get_cors_origins(),
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)
# Rate limiting (app/security.py): the slowapi limiter is registered as
# app.state.limiter for @limiter.limit decorators; RateLimitMiddleware
# enforces the per-route PATH_LIMITS table centrally.
app.state.limiter = limiter
app.add_middleware(RateLimitMiddleware)
app.include_router(auth.router)
app.include_router(livekit_webhook.router)
app.include_router(rooms.router)
app.include_router(transcripts.router)
app.include_router(sign_recognition.router)
app.include_router(transcription.router)
app.include_router(tts.router)
app.include_router(users.router)
app.include_router(voice_agent.router)
app.include_router(ai_status.router)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(request, exc: RateLimitExceeded):
    """slowapi decorator limits (@limiter.limit) answer 429 as JSON."""
    return JSONResponse(status_code=429,
                        content={"detail": f"Rate limit exceeded: {exc.detail}"})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc: Exception):
    """Unhandled errors answer JSON, never Starlette's plain-text
    'Internal Server Error'. HTTPException keeps its own (more
    specific) handler, so error responses elsewhere are unchanged."""
    logger.exception("Unhandled error on %s %s: %s",
                     request.method, request.url.path, exc)
    return JSONResponse(status_code=500,
                        content={"detail": "Internal server error"})


@app.get("/health")
def health():
    from app.services import sign_model_service
    return {
        "status": "ok", "service": "signspeak-telecom",
        "livekit_mode": "livekit" if livekit_service.livekit_configured() else "unconfigured",
        "sign_mode": ("trained-model" if sign_model_service.trained_available()
                      else "model-not-trained"),
        "whisper": "installed" if speech_service.whisper_available() else "not-installed",
        "tts": "piper" if __import__("importlib").util.find_spec("piper") else "not-installed",
    }


@app.get("/")
def root():
    return {"service": "SignSpeak Telecom API", "docs": "/docs", "health": "/health"}


# --- WebSocket signalling ---
_room_peers: dict[str, set[WebSocket]] = defaultdict(set)


@app.websocket("/ws/call/{room_id}")
async def call_ws(websocket: WebSocket, room_id: str, token: str = Query(default="")):
    # Auth first: a valid JWT is required. 4401 = unauthorised (custom close
    # code; 4401 mirrors HTTP 401). Never accept() an unauthenticated socket.
    payload = decode_token(token) if token else None
    sender = payload.get("sub") if payload else None
    if not sender:
        await websocket.close(code=4401)
        return
    # Room must exist (UUID or 6-char code) and be active, else 4404.
    db = SessionLocal()
    try:
        room = (
            db.query(Room).filter(Room.id == room_id).first()
            or db.query(Room).filter(Room.code == room_id.upper()).first()
        )
        if not room or not room.is_active:
            await websocket.close(code=4404)
            return
        room_key = room.id
    finally:
        db.close()
    await websocket.accept()
    _room_peers[room_key].add(websocket)
    try:
        await _broadcast(room_key, {"type": "join", "sender": sender, "payload": {}},
                         exclude=websocket)
        while True:
            raw = await websocket.receive_text()
            if len(raw.encode("utf-8")) > _WS_MAX_BYTES:
                await websocket.close(code=1009)  # message too big
                break
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue  # drop non-JSON
            # Validate: must be a JSON object with an allowed type; drop the rest.
            if not isinstance(msg, dict) or msg.get("type") not in _WS_ALLOWED_TYPES:
                continue
            # Never trust a client-supplied "sender": overwrite with the
            # authenticated user id. The client's "from" peer-id field is
            # left untouched — frontend WebRTC filtering uses it.
            msg["sender"] = sender
            # caption / sign / tts / transcript messages are relayed like signalling
            await _broadcast(room_key, msg, exclude=websocket)
    except WebSocketDisconnect:
        pass
    finally:
        _room_peers[room_key].discard(websocket)
        await _broadcast(room_key, {"type": "leave", "sender": sender, "payload": {}})


async def _broadcast(room_id: str, msg: dict, exclude=None):
    dead = []
    for ws in list(_room_peers.get(room_id, set())):
        if ws is exclude:
            continue
        try:
            await ws.send_text(json.dumps(msg))
        except Exception:
            dead.append(ws)
    for ws in dead:
        _room_peers[room_id].discard(ws)


# --- Live captions WebSocket: /ws/captions?token=<jwt> ---
# Client streams binary audio chunks (webm/opus or wav, e.g. from
# MediaRecorder). The server accumulates them and transcribes the
# buffer with faster-whisper every ~2s, sending:
#   {"type": "caption", "text", "confidence", "is_final": true,
#    "engine": "whisper-tiny"}          — a completed window
#   {"type": "error", "message": ...}   — model unavailable / decode error
#   (never fake captions: no text is sent that Whisper did not produce)
# JSON text messages are also accepted:
#   {"type": "text", "text": "..."}     — client transcription, echoed back
#     labelled engine "client-text-passthrough"
#   {"type": "flush"}                   — transcribe the buffer now
#   {"type": "reset"}                   — clear the audio buffer
# 64KB cap per message, same as /ws/call.
_CAPTION_WINDOW_SECONDS = 2.0
_CAPTION_MAX_BUFFER = 10 * 1024 * 1024


@app.websocket("/ws/captions")
async def captions_ws(websocket: WebSocket, token: str = Query(default="")):
    payload = decode_token(token) if token else None
    sender = payload.get("sub") if payload else None
    if not sender:
        await websocket.close(code=4401)
        return
    await websocket.accept()
    audio_buffer = bytearray()
    sent_text = ""
    last_window = asyncio.get_running_loop().time()
    try:
        while True:
            message = await websocket.receive()
            now = asyncio.get_running_loop().time()
            if "bytes" in message and message["bytes"] is not None:
                chunk = message["bytes"]
                if len(chunk) > _WS_MAX_BYTES:
                    await websocket.close(code=1009)
                    break
                audio_buffer += chunk
                if len(audio_buffer) > _CAPTION_MAX_BUFFER:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Audio buffer exceeded 10MB; send {\"type\": \"reset\"} "
                                   "to start a new utterance."}))
                    audio_buffer.clear()
                    sent_text = ""
                    continue
                if now - last_window < _CAPTION_WINDOW_SECONDS:
                    continue
            elif "text" in message and message["text"] is not None:
                raw = message["text"]
                if len(raw.encode("utf-8")) > _WS_MAX_BYTES:
                    await websocket.close(code=1009)
                    break
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(msg, dict):
                    continue
                if msg.get("type") == "text" and msg.get("text"):
                    await websocket.send_text(json.dumps({
                        "type": "caption", "text": str(msg["text"]),
                        "confidence": 0.0, "is_final": True,
                        "engine": "client-text-passthrough",
                        "note": "Text was supplied by the client; not transcribed by Whisper."}))
                    continue
                if msg.get("type") == "reset":
                    audio_buffer.clear()
                    sent_text = ""
                    last_window = now
                    continue
                if msg.get("type") != "flush":
                    continue  # unknown type: ignore
            else:
                continue
            # Transcribe the accumulated buffer (full buffer from the
            # start, so webm/wav headers stay valid) in a worker thread.
            if not audio_buffer:
                last_window = now
                continue
            last_window = now
            try:
                result = await speech_service.transcribe_audio_bytes_async(
                    bytes(audio_buffer))
            except Exception as exc:
                await websocket.send_text(json.dumps({
                    "type": "error", "message": str(exc)}))
                continue
            text = result.get("text", "")
            if text.startswith(sent_text):
                new_text = text[len(sent_text):].strip()
            else:
                new_text = text  # model revised its output; send it whole
            if new_text:
                sent_text = text
                await websocket.send_text(json.dumps({
                    "type": "caption", "text": new_text,
                    "confidence": result.get("confidence", 0.0),
                    "is_final": True, "engine": result.get("engine", "whisper-tiny")}))
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # never crash the socket on a bad frame
        logger.exception("captions websocket error: %s", exc)
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
