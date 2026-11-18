# Deployment

Rewritten 2026-11-18 (Priority 8). The old version of this file said LiveKit
credentials were optional and described a WebRTC P2P fallback — both are
wrong: **calls do not work without LiveKit, and there is no P2P fallback**
(the legacy `frontend/lib/webrtc.ts` was deleted).

## LiveKit is REQUIRED for calls

Without `LIVEKIT_URL` / `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET`,
`GET /api/rooms/{id}/token` returns 503 and no call can start. Use LiveKit
Cloud or a self-hosted server; for local dev run `livekit-server --dev`
(see `backend/livekit-dev.md`). Also configure the LiveKit webhook URL to
`POST /api/livekit/webhook` with `LIVEKIT_WEBHOOK_SECRET` so room/participant
events are verified (see `docs/API.md`).

## Frontend — Vercel
1. Import `frontend/` as the project root. Framework: Next.js 14.
2. Env: `NEXT_PUBLIC_API_URL=https://<backend-host>`, `NEXT_PUBLIC_WS_URL=wss://<backend-host>`.
3. Deploy. No server secrets in the frontend — never put LiveKit secrets or
   `SECRET_KEY` in frontend env vars.

> **MediaPipe assets are not in git.** The three binary files under
> `frontend/public/mediapipe/` (`hand_landmarker.task` and the two
> `.wasm` files) are git-ignored; after a fresh clone, restore them with
> `./frontend/scripts/restore-mediapipe-assets.sh` (run from the repo root
> after `npm install` in `frontend/`). It copies the wasm files from the
> `@mediapipe/tasks-vision` npm package and downloads the `.task` model
> from Google. Without them the build still passes, but in-browser sign
> detection fails at runtime.

## Backend — Docker (recommended)

Build context **must be `backend/`** (the directory containing the
Dockerfile):

```bash
docker build -t signspeak-backend backend/
docker run -p 8000:8000 --env-file backend/.env signspeak-backend
```

The image (`backend/Dockerfile`) is `python:3.12-slim`, runs as a non-root
user, installs the CPU-only torch wheel, and **downloads the three AI
models at build time** (faster-whisper tiny, Piper `en_US-lessac-medium`,
MediaPipe `hand_landmarker.task`) so a fresh container serves all three
pipelines immediately. The entrypoint (`backend/docker-entrypoint.sh`)
re-downloads any model that is still missing at container start (e.g. when
`models/` is mounted as an empty volume). `.dockerignore` keeps `.env`,
`*.db`, `.venv`, `models/` and `data/` out of the image.
Health check: `GET /health` (the Dockerfile wires it as `HEALTHCHECK`).

## Backend — Render / Railway (no Docker)
1. Import `backend/` as the project root. Python 3.12, then:
   ```bash
   pip install --index-url https://download.pytorch.org/whl/cpu torch
   pip install -r requirements.txt
   uvicorn app.main:app --host 0.0.0.0 --port $PORT
   ```
   (Plain `pip install torch` pulls the CUDA wheel — use the CPU index.)
2. Download the AI models once (or the endpoints 503 honestly):
   ```bash
   python -m app.services.speech_service --download   # faster-whisper tiny
   python -m app.services.tts_service --download      # Piper voice
   python -m app.services.vision_service --download   # hand_landmarker.task
   ```
3. Env (full list with comments: `backend/.env.example` — copy it to
   `.env`, never commit a real `.env`):
   `SECRET_KEY` (long random; **required** — the app refuses to start in
   production with a shipped default), `ENVIRONMENT=production`,
   `DATABASE_URL`, `FRONTEND_URL=https://<vercel-app>`,
   `LIVEKIT_URL` / `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` (**required**
   for calls), `LIVEKIT_WEBHOOK_SECRET`, `WHISPER_MODEL=tiny`,
   `SIGN_MODEL_PATH=app/ml/models/sign_lstm.pt` (trained weights; absent
   until you train — see `docs/TRAINING.md`),
   `SIGN_THRESHOLD_HIGH=0.85`, `SIGN_THRESHOLD_LOW=0.60`.
4. Health check: `GET /health` — it reports `livekit_mode`,
   `sign_mode`, `whisper` and `tts`, so a deploy can be verified at a glance.

## Database
SQLite by default (`signspeak.db`, created on startup). For Postgres set
`DATABASE_URL` to a `postgresql://...` URL — the `psycopg[binary]` driver
is in `requirements.txt` and `database.py` resolves the URL to
`postgresql+psycopg://`. Schema is managed with **Alembic**
(`backend/alembic/`, initial migration `761b2074d828`): run
`alembic upgrade head` after setting `DATABASE_URL`. Tables are still
created with SQLAlchemy `create_all` on startup for local SQLite dev, so
a fresh SQLite database works with no migration step. Full details in
`backend/docs/DATABASE.md`. Redis is optional: when `REDIS_URL` is set,
the per-room signal queue uses Redis (shared across workers); otherwise
it is in-memory. No other state uses Redis yet — see `DATABASE.md`.

## Notes
- Ports: backend 8000, frontend 3000, LiveKit 7880 (dev).
- CORS is an explicit allowlist (`http://localhost:3000`,
  `http://127.0.0.1:3000`, `FRONTEND_URL`, `CORS_ALLOWED_ORIGINS`) —
  never `*` with credentials.
- Recorded sign samples (`backend/data/sign_samples/`) and generated TTS
  audio (`backend/data/tts/`) are local user data: gitignored, and not
  part of any image.
