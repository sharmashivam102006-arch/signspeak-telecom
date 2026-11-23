# SignSpeak Telecom

**One Call. Two Languages. No Communication Barrier.**

AI-powered, real-time communication platform bridging Deaf / hard-of-hearing
and hearing people in the same video call.

- **Pipeline A — Sign → Text → Speech:** webcam → MediaPipe hand landmarks →
  temporal sign model → text → text-to-speech, delivered to the other participant.
- **Pipeline B — Speech → Text → Live Captions:** microphone → speech
  recognition → live captions for the Deaf participant, inside the call.

> Honesty note: no trained sign-language model ships with this repo.
> Sign recognition runs in an explicitly labelled **demo rule-based mode**
> (see § Implemented / Simulated / Pending). The PyTorch LSTM architecture,
> training pipeline and inference API are real and ready for trained weights.

## Quick start (local)

```bash
# Backend — http://localhost:8000, docs at /docs
cd backend
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set SECRET_KEY
uvicorn app.main:app --reload --port 8000
python -m pytest -v    # 9 tests

# Frontend — http://localhost:3000 (in a second terminal)
cd frontend
npm install
cp .env.example .env.local   # NEXT_PUBLIC_API_URL=http://localhost:8000
npm run dev
```

Open two browser windows/tabs, sign in (or **Continue as guest**), create a
room in one, copy the invite link (`/room/<code>`), join from the other.
Allow camera + microphone. Captions, sign demo, TTS and transcript all live
in the call page.

## Structure

```
signspeak-telecom/
  README.md            this file
  docs/API.md          full API reference
  docs/DEPLOYMENT.md   Vercel + Render/Railway + Postgres deployment
  docs/TRAINING.md     dataset format + how to train the real SignLSTM
  backend/             FastAPI + SQLAlchemy + WebSocket signalling
    app/main.py        app, CORS, /health, WS /ws/call/{room_id}
    app/routes/        auth, rooms, transcripts, sign_recognition,
                       transcription, tts, users
    app/services/      livekit, vision/landmark, sign_model, speech, tts,
                       transcript
    app/ml/            models/sign_lstm.py, preprocessing, training, inference
    tests/test_api.py  9 passing tests
    requirements.txt   light deps only; torch/mediapipe/whisper optional
    schema.sql  Dockerfile  .env.example
  frontend/            Next.js 14 App Router + TypeScript + Tailwind
    app/page.tsx       landing — "Communication Without Barriers."
    app/login, app/signup, app/dashboard
    app/room/[id]/    video call: WebRTC P2P, captions, AI panel,
                       confidence + correction, transcript
    lib/               api, webrtc, sign (MediaPipe), speech, tts, store
    components/        Logo, Navbar, VideoTile, CaptionBar,
                       TranscriptPanel, ConfidenceCard, AIPanel, CallControls
```

## Design

Dark-first: background `#080B14`, secondary `#101827`, cyan `#22D3EE`,
violet `#8B5CF6`, success `#22C55E`, text `#F8FAFC`, muted `#94A3B8`.

## API at a glance

| Area | Endpoints |
|---|---|
| Health | `GET /health`, `GET /` |
| Auth | `POST /api/auth/register`, `POST /api/auth/login`, `POST /api/auth/guest`, `GET /api/auth/me` |
| Rooms | `POST /api/rooms`, `GET /api/rooms`, `GET /api/rooms/{id}`, `POST /api/rooms/{id}/join`, `POST /api/rooms/{id}/signal` |
| Sign AI | `GET /api/ai/sign/vocabulary`, `POST /api/ai/sign/predict` |
| Speech | `POST /api/ai/transcribe`, `POST /api/ai/transcribe/audio` |
| TTS | `POST /api/ai/tts`, `GET /api/ai/tts/voices` |
| Transcripts | `GET/POST /api/rooms/{id}/transcripts`, `GET /api/rooms/{id}/transcripts/export` |
| Signalling | `WS /ws/call/{room_id}?token=…` — offer/answer/ice-candidate/join/leave/caption/sign/tts/transcript |

Full reference: [docs/API.md](docs/API.md). Interactive docs: `/docs` when running.

## Implemented / Simulated / Pending

### Implemented and verified
- JWT + bcrypt auth, guest/demo mode, protected dashboard routes.
- Rooms: unique 6-char code + UUID, invite link, join, invalid-room 404.
- WebRTC P2P video calls via backend WebSocket signalling (+ REST fallback),
  mic/cam/end controls, participant + connection status.
- Live captions via Web Speech API (interim + final, deduped); TTS via
  SpeechSynthesis with mute/volume/voice, duplicate suppression.
- Transcript panel: sign→speech (violet) vs speech→text (cyan), timestamps,
  direction, copy / clear / export `.txt`; consent-gated server storage.
- Confidence system: thresholds 85% / 60% (configurable), Accept / Correct /
  Repeat / Ignore; corrections stored only with explicit consent.
- Landmark normalisation, 30-frame sequence buffer, prediction smoothing,
  2-second duplicate suppression.
- Backend tests: **33 passed, 1 skipped** (verified 17 Nov 2026 after the
  real AI pipelines landed — 24 from before plus 9 new pipeline tests:
  sign status/collect/train lifecycle, MediaPipe extract, real Whisper
  transcription, captions WebSocket, AI status summary, voice-agent audio
  pipeline, and the voice-status regression test; the 1 skip is the
  voice-agent LiveKit join, which cannot complete in this sandbox —
  see `backend/docs/AI_PIPELINES.md`);
  frontend `npm run build` passes; live smoke test: `/health`
  reports `livekit_mode: "livekit"`, vocabulary returns 6 signs, guest
  login returns a JWT, and a real LiveKit token is issued per room.
- **LiveKit calling is real** (since 17 Nov 2026): self-hosted LiveKit
  server v1.13.11 (`livekit/livekit-server` + `livekit.yaml`), real JWTs
  from `GET /api/rooms/{id}/token`, participant join/leave/end endpoints
  and webhook (`backend/livekit-dev.md`). Two-browser-session test verified
  room creation/join, server-side presence of both participants, LiveKit
  signalling connect, and mute/camera/end-call controls — but remote
  video could **not** be verified in this sandbox: it blocks UDP sockets
  and intercepts non-loopback TCP, so ICE never completes here. Verify
  media on a normal machine/network.

### Simulated — clearly labelled in the UI and API responses
- **Sign recognition: `mode: "demo-rule-based"`.** Deterministic hand-pose
  heuristics (open palm → Hello, index → Yes, fist → No, V → Help,
  thumb-up → Thank You, pinky → Emergency), confidence capped at 0.78,
  always `requires_confirm`. The six demo-sign buttons in the call UI run
  through the same confidence/correction pipeline at a fixed 82% and are
  labelled Demo. These are **not** model predictions.
- **Speech-to-text in the browser** uses the Web Speech API where the
  browser supports it; otherwise the UI says so instead of faking captions.
- **TTS** is browser SpeechSynthesis (`audio_url: null,
  engine: browser-speech-synthesis`); no audio file is synthesised server-side.

### Pending / needs keys, data or a trained model
- **Trained SignLSTM weights:** architecture (`SignLSTM`, 63-dim input,
  seq 30 → 6 classes), training script and dataset format exist
  ([docs/TRAINING.md](docs/TRAINING.md)). No public labelled dataset was
  bundled; train on your own recordings, set `SIGN_MODEL_PATH`, and the
  same `/api/ai/sign/predict` endpoint serves the real model.
- **Whisper server ASR:** `faster-whisper` is not installed; audio upload
  returns 503 with instructions. Install it and set `WHISPER_MODEL` to enable.
- **LiveKit Cloud / production keys:** local dev uses the self-hosted
  server with dev keys (`devkey` / `secret`). For production, point
  `LIVEKIT_URL/API_KEY/API_SECRET` at LiveKit Cloud (see
  `backend/livekit-dev.md` §5).
- **Production database:** SQLite by default; set `DATABASE_URL` to Postgres
  for deployment. See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).
- No recordings are stored by default; transcripts only with consent.

## Security & privacy
- Secrets only via environment variables; never exposed to the frontend.
- `SECRET_KEY`: the app refuses to start with a shipped default secret when
  `ENVIRONMENT=production` (set a long random `SECRET_KEY` in `.env`); in
  development it logs a warning instead. CORS is an explicit allowlist —
  never `*` with credentials.
- Input validation via Pydantic; protected APIs via Bearer JWT.
- No video/audio recording by default; clear AI-processing indicators in the call UI.
