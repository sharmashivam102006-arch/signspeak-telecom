# SignSpeak Telecom — Backend

FastAPI backend for the AI-powered bidirectional sign-language / speech
communication platform. Frontend lives in `../frontend` (separate app).

## Quick start

```bash
cd backend
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set SECRET_KEY at minimum
uvicorn app.main:app --reload --port 8000
# Docs: http://localhost:8000/docs   Health: http://localhost:8000/health
```

Docker (build context must be this directory):
`docker build -t signspeak-backend . && docker run -p 8000:8000 --env-file .env signspeak-backend`
See `Dockerfile` — it downloads the Whisper/Piper/MediaPipe models at build
time and runs as a non-root user. Full deploy notes: `../docs/DEPLOYMENT.md`.
CI runs the backend tests + frontend build/tests on every push
(`../.github/workflows/ci.yml`).

## API summary

| Area | Endpoint |
|---|---|
| Health | `GET /health`, `GET /` |
| Auth | `POST /api/auth/register`, `POST /api/auth/login`, `GET /api/auth/me`, `POST /api/auth/guest` |
| Rooms | `POST /api/rooms`, `GET /api/rooms`, `GET /api/rooms/{id\|code}`, `POST /api/rooms/{id}/join`, `POST /api/rooms/{id}/leave`, `GET /api/rooms/{id}/participants`, `POST /api/rooms/{id}/end`, `GET /api/rooms/{id}/token`, `POST|GET /api/rooms/{id}/signal` |
| Transcripts | `GET|POST /api/rooms/{id}/transcripts`, `GET /api/rooms/{id}/transcripts/export` |
| Sign AI | `GET /api/ai/sign/vocabulary`, `GET /api/ai/sign/status`, `POST /api/ai/sign/predict`, `POST /api/ai/sign/extract`, `POST /api/ai/sign/collect`, `GET /api/ai/sign/samples`, `POST /api/ai/sign/train`, `GET /api/ai/sign/train/{job_id}` |
| Speech AI | `POST /api/ai/transcribe`, `POST /api/ai/transcribe/audio` (multipart), `WS /ws/captions?token=<jwt>` (live Whisper captions) |
| TTS | `POST /api/ai/tts`, `GET /api/ai/tts/voices`, `GET /api/ai/tts/audio/{audio_id}` |
| Voice agent | `POST /api/ai/rooms/{id}/voice/start`, `POST /api/ai/rooms/{id}/voice/speak`, `POST /api/ai/rooms/{id}/voice/stop`, `GET /api/ai/rooms/{id}/voice/status` |
| AI status | `GET /api/ai/status` |
| LiveKit webhook | `POST /api/livekit/webhook` (signature-verified) |
| Preferences | `GET|PUT /api/users/me/preferences` |
| Signalling WS | `WS /ws/call/{room_id}?token=<jwt>` — relays `caption/sign/tts/transcript` between peers in the room |

Auth: JWT Bearer (24h), bcrypt password hashing. Guest mode issues a
guest JWT with no password. Protected routes require Bearer token.
Security: `SECRET_KEY` must be a long random value — the app refuses to
start with a shipped default when `ENVIRONMENT=production` (development
only logs a warning). CORS is an explicit allowlist (localhost + `FRONTEND_URL`
+ `CORS_ALLOWED_ORIGINS`), never `*` with credentials. The WebSocket
requires a valid JWT and an existing active room; transcript storage is
consent-gated and creator-only. Participant membership is tracked in
`room_participants` (see `livekit-dev.md`).

## Real-time

Media runs on **real LiveKit** (see `livekit-dev.md`): run
`livekit-server --dev` (ws://localhost:7880, dev keys devkey/secret) and
`GET /api/rooms/{id}/token` issues a real LiveKit JWT (room = room code,
identity = user id, creator gets `room_admin`). With LiveKit
unconfigured the token endpoint returns **503** — there is no silent
P2P fallback for media. The legacy `/ws/call` WebSocket only relays
captions/sign messages between peers. Participant membership is tracked
in the `room_participants` table (join/leave/end + LiveKit webhook
`POST /api/livekit/webhook`).

## AI — what is real (updated 2026-11-18; the old "stub / demo-only" text below was stale)

- **Sign recognition — real pipeline, model NOT trained yet.**
  MediaPipe landmark extraction is real (server `POST /api/ai/sign/extract`
  and the browser HandLandmarker, 21 landmarks, 1 hand), and the PyTorch
  `SignLSTM` (63-in, seq 30, 6-class) architecture + training pipeline are
  real (`app/ml/models/sign_lstm.py`, `app/ml/training/train.py`). But no
  trained weights exist yet: with no `SIGN_MODEL_PATH` weights,
  `POST /api/ai/sign/predict` answers **503 `model-not-trained`** and
  `GET /health` reports `sign_mode:"model-not-trained"`. Deterministic
  hand-pose heuristics (open palm → Hello, index only → Yes, fist → No,
  V → Help, thumb-up → Thank You, pinky → Emergency) run **only** when the
  caller passes `demo:true`; their fixed confidences are capped at 0.78 and
  the response is labelled `mode:"demo-rule-based"` — the live path never
  surfaces them as recognition. Once weights exist, predict returns real
  softmax confidences with `mode:"trained-model"`. Duplicate predictions
  within 2 s are suppressed. Thresholds: high 0.85 / low 0.60 (env
  `SIGN_THRESHOLD_HIGH` / `SIGN_THRESHOLD_LOW`). See `../docs/TRAINING.md`.
- **Speech — Whisper ASR is real.** `faster-whisper` (tiny, int8, CPU) is
  installed; `POST /api/ai/transcribe/audio` returns genuine transcriptions
  with confidence scores, and `WS /ws/captions` streams live captions from
  ~2 s audio chunks. Without faster-whisper those endpoints 503.
  `POST /api/ai/transcribe` still exists for client-transcribed text (e.g.
  browser Web Speech API) and labels it `source:"client-stt"` — it is not
  Whisper output.
- **TTS — Piper is real.** `POST /api/ai/tts` synthesises genuine WAV audio
  with Piper (`en_US-lessac-medium`, 22050 Hz) and returns
  `{audio_url:"/api/ai/tts/audio/<id>", engine:"piper", duration_s}`; text
  is capped at 500 chars. The "SignSpeak Voice" agent
  (`/api/ai/rooms/{id}/voice/*`) joins the LiveKit room as a participant
  and publishes that audio so the remote participant hears it. Without
  Piper installed, TTS endpoints 503.

## Training a real model

1. Record MediaPipe landmark sequences per sign — in the app via the Train
   Signs panel (`POST /api/ai/sign/collect`, saved as JSON under
   `SIGN_SAMPLES_DIR/<label>/<uuid>.json`), at least 5 samples per sign
   from 2–3 different signers. Check counts with
   `GET /api/ai/sign/samples`.
2. Train via the API (`POST /api/ai/sign/train`, poll
   `GET /api/ai/sign/train/{job_id}`) or the CLI with a CSV dataset
   (`sample_id,label,frame,landmark_0_x,...,landmark_20_z` — see
   `app/ml/training/train.py` docstring):
   `pip install --index-url https://download.pytorch.org/whl/cpu torch`
   then
   `python -m app.ml.training.train --data data/signs.csv --epochs 30 --out app/ml/models/sign_lstm.pt`
3. Weights land at `SIGN_MODEL_PATH` (default, relative to `backend/`:
   `app/ml/models/sign_lstm.pt`).
4. Restart the server; `/api/ai/sign/predict` will use the weights
   (`mode: "trained-model"`, `GET /health` reports
   `sign_mode:"trained-model"`) when a sequence is supplied.

Without `--data`, the training script prints instructions and exits 0 —
it never fabricates data. Full details: `../docs/TRAINING.md`.

## Database / migrations

SQLite by default (`DATABASE_URL=sqlite:///./signspeak.db`); set
`DATABASE_URL` to a Postgres URL for production. Tables are created
with SQLAlchemy `create_all` on startup; `schema.sql` documents the
schema (Alembic-style migration source). For production, generate
Alembic migrations from the models in `app/models.py`.

## Tests

```bash
python -m pytest -v
```
Backend: `tests/test_api.py` (34 tests) uses a separate
`test_signspeak.db` and covers register/login, guest, room create/join +
invalid-room 404, sign vocabulary + demo predict + train-job lifecycle,
TTS, transcribe/transcripts/export, preferences, and WebSocket
signalling. The voice-agent lifecycle test is skipped in sandboxed
environments (it needs a reachable LiveKit server to join a room).
Frontend: `npm test` in `../frontend` runs Vitest unit tests for the
sign-smoothing / duplicate-suppression logic (`lib/sign.ts`), caption
helpers (`lib/captions.ts`) and the API client (`lib/api.ts`).
