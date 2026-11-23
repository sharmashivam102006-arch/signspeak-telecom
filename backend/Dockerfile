# SignSpeak Telecom — backend image.
#
# Build context MUST be backend/ (the directory containing this file):
#   docker build -t signspeak-backend backend/
#   docker run -p 8000:8000 --env-file backend/.env signspeak-backend
#
# Design decisions (Priority 8, 2026-11-18):
# - python:3.12-slim, non-root user, no build tools in the final image.
# - CPU-only torch: the default PyPI wheel is CUDA and ~800 MB larger.
# - The three AI models are downloaded AT BUILD TIME from public, stable
#   URLs (Hugging Face + Google Cloud Storage — all reachable, verified
#   2026-11-18), so a fresh container serves Whisper/Piper/MediaPipe
#   immediately. If a download fails the build fails loudly; the image
#   never ships a container that 503s every AI endpoint. The entrypoint
#   re-downloads any model that is still missing at container start
#   (e.g. when models/ is mounted as an empty volume).
# - .dockerignore excludes .env, *.db, .venv, models/ and data/ from the
#   build context, so no secret, database or 2.1 GB venv is baked in.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/appuser/.cache/huggingface

WORKDIR /app

# System libraries MediaPipe / OpenCV need at runtime (no build tools):
#   libgl1        — MediaPipe needs libGL.so.1
#   libglib2.0-0  — OpenCV-headless + MediaPipe need libglib/libgthread
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 \
      libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r appuser && useradd -r -g appuser -d /home/appuser -s /usr/sbin/nologin appuser \
    && mkdir -p /home/appuser/.cache /app/models /app/data/tts \
    && chown -R appuser:appuser /home/appuser /app

# Dependencies first (layer cache): requirements.txt changes rarely.
COPY requirements.txt .
# CPU-only torch wheel — plain `pip install torch` pulls the CUDA wheel.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY schema.sql ./
COPY docker-entrypoint.sh ./
RUN chmod +x docker-entrypoint.sh

# Download the AI models at build time (see header comment). Each service
# has a `--download` entry point that writes into /app/models.
RUN python -m app.services.speech_service --download \
    && python -m app.services.tts_service --download \
    && python -m app.services.vision_service --download

RUN chown -R appuser:appuser /app /home/appuser
USER appuser

EXPOSE 8000

# /health reports livekit_mode / sign_mode / whisper / tts; a 200 means
# the API is up (AI pipelines report their own state in the body).
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
