"""Configuration for SignSpeak Telecom backend.

All secrets come from environment variables (see .env.example).
Nothing here is a real credential.
"""
import logging
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

ENVIRONMENT = os.getenv("ENVIRONMENT", "development").strip().lower()

_DEFAULT_SECRETS = {
    "dev-only-change-me-in-production",
    "change-me-to-a-long-random-secret",
}
SECRET_KEY = os.getenv("SECRET_KEY", "dev-only-change-me-in-production")
if SECRET_KEY in _DEFAULT_SECRETS:
    if ENVIRONMENT == "production":
        raise RuntimeError(
            "SECRET_KEY is still a shipped default and ENVIRONMENT=production. "
            "Set a long random SECRET_KEY in the environment before starting in production."
        )
    logger.warning(
        "SECRET_KEY is a shipped default — fine for local development only. "
        "Set a long random SECRET_KEY (and ENVIRONMENT=production) before deploying."
    )
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_HOURS = int(os.getenv("ACCESS_TOKEN_EXPIRE_HOURS", "24"))
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./signspeak.db")

# REDIS_URL: optional. When set (e.g. redis://localhost:6379/0) the
# state store (app.services.state_store) uses Redis for per-room signal
# queues and training-job status instead of process-local memory, so
# state survives restarts and is shared across workers. When empty
# (default) the in-memory backend is used — fine for local dev.
REDIS_URL = os.getenv("REDIS_URL", "").strip()

LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "secret")
# LiveKit access-token lifetime. The livekit-api SDK defaults to 6h;
# this makes the TTL explicit and configurable (same default).
LIVEKIT_TOKEN_TTL_HOURS = int(os.getenv("LIVEKIT_TOKEN_TTL_HOURS", "6"))
# Dev defaults above are the keys of a local `livekit-server --dev`
# (see backend/livekit-dev.md). PRODUCTION MUST replace them with real
# LiveKit Cloud / self-hosted credentials -- never ship devkey/secret.
# Optional secret used to verify LiveKit webhook signatures; when empty,
# webhook verification falls back to LIVEKIT_API_SECRET.
LIVEKIT_WEBHOOK_SECRET = os.getenv("LIVEKIT_WEBHOOK_SECRET", "")

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en")
# Local cache dir for the faster-whisper model (ctranslate2 format).
# Default: backend/models/whisper-<model>. Download with:
#   python -m app.services.speech_service --download
WHISPER_MODEL_DIR = os.getenv(
    "WHISPER_MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "models", f"whisper-{WHISPER_MODEL}"))
# CPU threads for Whisper inference. 0 = auto: the container's CPU quota
# (cgroup cpu.max) when set, else os.cpu_count(), capped at 4. Letting
# ctranslate2 spawn one thread per *host* core on a quota-limited
# container (e.g. a small Render instance) oversubscribes the CPU and
# makes every transcription many times slower.
WHISPER_CPU_THREADS = int(os.getenv("WHISPER_CPU_THREADS", "0"))
# Beam size: 1 = greedy decoding, several times faster on CPU than
# faster-whisper's default of 5, with a small accuracy cost on tiny.
WHISPER_BEAM_SIZE = max(1, int(os.getenv("WHISPER_BEAM_SIZE", "1")))
# Live captions (/ws/captions): utterances are cut at a pause, or at
# CAPTION_MAX_SEGMENT_S if the speaker keeps talking. If captions fall
# more than CAPTION_MAX_LAG_S behind the speaker (server too slow for
# real-time Whisper), the socket sends {"type":"error","code":"too-slow"}
# so the client switches to browser captions instead of waiting forever.
CAPTION_MAX_SEGMENT_S = float(os.getenv("CAPTION_MAX_SEGMENT_S", "8"))
CAPTION_MAX_LAG_S = float(os.getenv("CAPTION_MAX_LAG_S", "12"))
# Seconds the caption socket waits for the Whisper model to load before
# reporting too-slow (first use after a deploy / cold start).
CAPTION_MODEL_LOAD_TIMEOUT_S = float(os.getenv("CAPTION_MODEL_LOAD_TIMEOUT_S", "45"))

# MediaPipe HandLandmarker model (Tasks API). Download with:
#   python -m app.services.vision_service --download
MEDIAPIPE_MODEL_PATH = os.getenv(
    "MEDIAPIPE_MODEL_PATH", os.path.join(os.path.dirname(__file__), "..", "models", "hand_landmarker.task"))

SIGN_THRESHOLD_HIGH = float(os.getenv("SIGN_THRESHOLD_HIGH", "0.85"))
SIGN_THRESHOLD_LOW = float(os.getenv("SIGN_THRESHOLD_LOW", "0.60"))
SIGN_SEQUENCE_LENGTH = int(os.getenv("SIGN_SEQUENCE_LENGTH", "30"))
SIGN_MODEL_PATH = os.getenv("SIGN_MODEL_PATH", "app/ml/models/sign_lstm.pt")
# Directory holding user-recorded landmark sequences (JSON, one file per
# sample) used to train the sign model. No synthetic data is stored here.
SIGN_SAMPLES_DIR = os.getenv(
    "SIGN_SAMPLES_DIR", os.path.join(os.path.dirname(__file__), "..", "data", "sign_samples"))
SIGN_SAMPLES_PER_LABEL_MAX = int(os.getenv("SIGN_SAMPLES_PER_LABEL_MAX", "200"))

TTS_ENGINE = os.getenv("TTS_ENGINE", "piper")
# Piper voice: "<lang>_<REGION>-<name>-<quality>", e.g. en_US-lessac-medium.
PIPER_VOICE = os.getenv("PIPER_VOICE", "en_US-lessac-medium")
# Directory holding Piper .onnx voice files (downloaded on demand).
PIPER_VOICE_DIR = os.getenv(
    "PIPER_VOICE_DIR", os.path.join(os.path.dirname(__file__), "..", "models", "piper"))
# Directory holding synthesised TTS WAV files served by /api/ai/tts/audio/<id>.
TTS_AUDIO_DIR = os.getenv(
    "TTS_AUDIO_DIR", os.path.join(os.path.dirname(__file__), "..", "data", "tts"))
TTS_MAX_CHARS = int(os.getenv("TTS_MAX_CHARS", "500"))

# Voice agent (LiveKit "SignSpeak Voice" participant that speaks TTS into
# the room). Disable to run the API without the agent feature.
VOICE_AGENT_ENABLED = os.getenv("VOICE_AGENT_ENABLED", "true").strip().lower() in ("1", "true", "yes")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

# Admin allowlist for privileged operations (e.g. POST /api/ai/sign/train).
# Comma-separated env vars; empty by default (nobody is an admin).
# Example: ADMIN_EMAILS=ops@example.com,ADMIN_USER_IDS=<uuid>
ADMIN_EMAILS = {
    e.strip().lower() for e in os.getenv("ADMIN_EMAILS", "").split(",") if e.strip()
}
ADMIN_USER_IDS = {
    i.strip() for i in os.getenv("ADMIN_USER_IDS", "").split(",") if i.strip()
}

# CORS allowlist. Never "*": with allow_credentials=True browsers reject
# a wildcard origin, and it would expose credentialed APIs to any site.
# Extra origins via CORS_ALLOWED_ORIGINS (comma-separated).
CORS_ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("CORS_ALLOWED_ORIGINS", "").split(",") if o.strip()
]


def get_cors_origins() -> list:
    origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
    for origin in [FRONTEND_URL, *CORS_ALLOWED_ORIGINS]:
        origin = (origin or "").strip().rstrip("/")
        if origin and origin != "*" and origin not in origins:
            origins.append(origin)
    return origins
