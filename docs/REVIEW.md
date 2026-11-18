# SignSpeak Telecom — Code Review (17 Nov 2026)

> **Status update (17 Nov 2026, after review): all 10 P0 items are fixed
> and verified — backend 15/15 tests pass (9 original + 6 new), frontend
> `npm run build` passes. See each P0 section's fix note in the fix
> summary below; the P1/P2 items remain open and are listed in suggested
> order at the end.**

Review of the whole repo as built: `backend/` (FastAPI, ~1,500 LOC Python),
`frontend/` (Next.js 14, ~1,400 LOC TS/TSX), `docs/`. Method: read every
source file, traced the two pipelines end to end, re-ran the backend tests
(9 passed) and a live smoke test. No redesign of working components; fixes
below are scoped to the defect.

## Verdict

The scaffolding, auth, rooms and demo labelling are honest and solid. But
the **core success criterion is not actually met end to end**: the live
sign path, the dashboard room flow, and the cross-peer pipelines are broken
in ways that only show up when two real people use the app. Those are P0.
Security has three P0s (CORS, unauthenticated signalling, transcript
consent). Everything else is P1/P2.

---

## P0 — Broken core flow or security (fixing now)

### P0-1. Dashboard "Start New Call" creates a room that does not exist
`frontend/app/dashboard/page.tsx:startCall` generates a local `SS-XXXX` id,
POSTs `{id}` (backend `RoomCreate` only accepts `name`, so the id is
ignored), creates a *different* server room with a 6-char code, then
navigates to `/room/<local-id>`. The invite link, copy-link and join path
all use the local id. Two users can never meet in the same server room
from this flow.
Fix: use the returned `room.code` and navigate to `/room/<code>`.

### P0-2. Invite links 404
`backend/app/services/livekit_service.py:room_url` returns
`{FRONTEND_URL}/call/{code}`. The frontend has no `/call` route; the call
page is `/room/[id]`. Every invite link from the API is dead.
Fix: `/room/{code}`.

### P0-3. Live sign recognition never returns a prediction
Three stacked mismatches in `frontend/lib/sign.ts` ↔
`backend/app/services/sign_model_service.py`:
- Frontend buffers **flattened 63-float frames** and sends them as
  `landmarks` (a 30×63 matrix). Backend `landmarks_valid` requires exactly
  21 points → `mode: "no-hand"`, prediction `null`.
- Backend expects `sequence` as frames×21×3; the frontend never sends
  `sequence` at all.
- Frontend reads `p.sign` / `p.confidence`; the backend returns
  `{prediction, confidence, mode}` — so even a success would surface as
  `undefined`.
- Bonus: `_trained_predict` feeds **raw unnormalised** landmarks to the
  model while training (`train.py`) uses normalised features
  (`sequence_to_features`) — a trained model would get a different
  distribution from the one it was trained on. It also reloads the weights
  from disk on every request and calls `torch.load` without
  `weights_only=True` (arbitrary-code execution if a weights file is tampered with).
Fix: buffer raw 21×3 landmarks, send `sequence`, map the response fields,
normalise in `_trained_predict`, cache the model, `weights_only=True`.

### P0-4. The bidirectional pipelines do not cross the call
This is the product's core promise, and it is local-only on both sides:
- Speech captions: `room/[id]/page.tsx` transcribes the **local**
  microphone and shows the captions **locally**. The Deaf participant
  needs the *other* person's speech; they never receive it — captions and
  sign messages are never sent over the WebSocket, and `CallClient` has no
  handler for incoming `caption`/`sign`/`tts` messages (the backend
  broadcasts them, the frontend ignores them).
- Sign→speech: `speakText` plays via the **local** `speechSynthesis`.
  The footer copy ("Generated speech is played for the other participant")
  and `AIPanel` ("spoken aloud so the hearing participant hears them")
  are false. No TTS audio is injected into the WebRTC audio track.
- The "Real AI" badge is set merely because MediaPipe JS loaded, while
  the backend mode is `demo-rule-based`.
Fix (P0 part): relay `caption`/`sign` messages over WS and render remote
ones; label sign status from the backend `mode`, not from MediaPipe load.
Fix (P1, real work): inject synthesised speech into the outgoing audio
track (e.g. server TTS → audio element → `MediaStreamDestination` mixed
into the peer connection) or send the text and speak it on the remote
side with an explicit "remote TTS" indicator.

### P0-5. WebSocket signalling is unauthenticated and spoofable
`backend/app/main.py:call_ws` accepts any connection, including with no
or an invalid token (`sender = "anonymous"`), never checks the room
exists, and `msg.setdefault("sender", sender)` trusts a client-supplied
`sender` — anyone can impersonate any user in any room, including rooms
that do not exist. No message-size limit, no type allowlist.
Fix: require a valid JWT, verify the room exists and is active, overwrite
`sender` with the token `sub`, allowlist message types, 64 KB cap.
Frontend part: `lib/webrtc.ts` connects with `?peer=` only and never
sends the token; it also sends ICE as type `"ice"` while the backend
schema/docs say `"ice-candidate"`.

### P0-6. CORS: `*` + credentials
`backend/app/main.py` sets `allow_origins=[..., "*"]` with
`allow_credentials=True`. Browsers reject that combination and it is
insecure if ever honoured. Fix: explicit allowlist from env.

### P0-7. Transcript consent is claimed but not enforced
`transcript_service.py` and the README say transcripts are stored "ONLY
when the user consents". `TranscriptCreate` has no consent field and
`routes/transcripts.py:add_transcript` stores unconditionally. Worse,
**any** authenticated user can list/export **any** room's transcripts —
no membership check (there is no participant model at all; `join_room`
records nothing). Fix: require `consent` or the stored
`store_transcripts` preference; restrict reads/writes to the room
creator until a participant model exists.

### P0-8. Duplicate suppression is global across all users
`sign_model_service.py` keeps one module-level `_last_prediction` /
`_last_prediction_time` for the whole process. One user's "Hello"
suppresses another user's "Hello" for 2 s, and it is not thread-safe.
Fix: scope the state per caller (user id / anonymous key).

### P0-9. Unbounded, unauthenticated REST signal store
`routes/rooms.py` `_signal_store` is an in-memory dict with no auth, no
cap and no expiry; `GET` pops the whole queue (racy). Fix: require auth,
cap at 100 per room, note it is single-process/ephemeral.

### P0-10. No payload size limits
`SignPredictRequest.sequence` is unbounded; `/transcribe/audio` does
`await file.read()` with no cap. A single huge request can OOM the
single-process server. Fix: 21 landmarks / 30 frames; 10 MB audio cap.

---

## P1 — Real gaps, not just polish

1. **WebRTC glare / no polite negotiation.** Both peers send an offer on
   `join` (`webrtc.ts:createOffer` on every join, `makingOffer` is only a
   local flag). Two simultaneous offers → `setRemoteDescription` failures.
   Fix: one side (e.g. second joiner) is the offerer, or implement
   polite/impolite negotiation with rollback.
2. **STUN-only, no TURN.** `webrtc.ts` uses only Google STUN. Calls fail
   on symmetric NATs / corporate networks. Add a TURN server config.
3. **Speech recognition stops silently.** `room/[id]` starts the
   recogniser once; Chrome ends it on silence/errors and `onEnd` never
   restarts it. Captions die mid-call with no visible state. Fix:
   auto-restart with backoff while the call is live.
4. **Room page never validates the room.** It opens the call UI for any
   id, including typos, and only fails opaquely at signalling. Fix: GET
   `/api/rooms/{id}` first; show "room not found" (the dashboard join
   path has the same gap — see P0-1 fix, which validates first).
5. **Settings are two sources of truth.** Dashboard settings live only
   in `localStorage` (`lib/store.ts`); the backend `/api/users/me/preferences`
   endpoints exist but no frontend code ever calls them — thresholds,
   language and TTS prefs do not follow the user across devices, and
   `PreferencesOut` omits `sign_threshold_high/low` that the model stores.
   Fix: sync settings to the backend on change; add the missing fields.
6. **Server TTS / Whisper / LiveKit are stubs, by design but untested.**
   `speech_service.transcribe_audio_bytes` creates a new `WhisperModel`
   per call (minutes of load per request); `vision_service.extract_hand_landmarks`
   creates a new MediaPipe `Hands` per frame. Both need module-level
   lazy singletons when enabled. No integration test covers any of them.
7. **No participant model.** Spec requires room participants/session
   metadata. `Room` has only `creator_id`; join records nothing, so
   presence, "who is in the call", end-room and transcript access control
   cannot be built correctly. Add a `RoomParticipant` table.
8. **JWT in `localStorage` + token in WS query string.** XSS can steal
   the token; query strings are logged by proxies. Acceptable for a demo,
   but the production move is httpOnly cookies for the API and a
   short-lived WS ticket (or `Sec-WebSocket-Protocol` header) instead of
   `?token=`.
9. **Guest accounts accumulate forever.** Every `/api/auth/guest` call
   creates a permanent `User` + `UserPreferences` row; there is no expiry
   or cleanup job, and no rate limit on register/guest/login (brute-force
   + enumeration: register returns "Email already registered").
10. **No end-room flow.** `is_active` is checked on join but nothing
    ever sets it false; no `POST /rooms/{id}/end`, no WS room teardown.
11. **Dashboard rooms list never renders.** `api("/api/rooms")` returns
    a bare array; the dashboard reads `d.rooms ?? []` → always empty.
    (Fixed with P0-1.)

## P2 — Quality, hygiene, unnecessary complexity

- **No backend `.gitignore`.** `signspeak.db`, `test_signspeak.db`,
  `.venv/`, `__pycache__/` are in the tree and would be committed.
  Dockerfile `COPY . .` also copies them into the image; no
  `.dockerignore`, no `HEALTHCHECK`, runs as root, no `--workers`.
- **Dead / duplicated code.** `vision_service.extract_hand_landmarks` is
  never called by any route; `landmark_service.py` is a thin re-export of
  `normalize.py`; `ml/inference/predict.py` duplicates
  `sign_model_service.predict`; `tts_service.VOICES` is a hardcoded list
  presented as voices; `speech_service.transcribe_text` labels its output
  `engine: "whisper-..."` when Whisper is not installed (misleading —
  it is a passthrough). `SignLSTM`'s no-torch fallback class, the
  `get_optional_user` dual path, and the LiveKit-stub + P2P dual path add
  surface without behaviour. Remove or wire up; do not add more.
- **Validation gaps.** `TranscriptCreate.direction` is a free string;
  `PreferencesUpdate.role` is a free string; `RoomJoin.code` is not
  checked against the room on the GET path; `UserLogin.password` has no
  length limit (fine) but login has no rate limit (P1-9).
- **Training script is a skeleton.** `train.py` has no train/val split,
  no label validation (`VOCABULARY.index(label)` raises unhandled on a
  bad label), no seed, no metrics, and saves bare `state_dict` with no
  metadata (vocab, seq len, normalisation version) — a weights file
  cannot be validated on load.
- **Tests are thin and order-dependent.** 9 tests, all happy-path plus
  one 404. No security tests (auth bypass, cross-user transcript read,
  WS spoofing), no frontend tests at all, no CI. Test DB is a file
  (`test_signspeak.db`) with `drop_all` at import time — dangerous
  outside a test run.
- **Docs drift.** `docs/API.md` says signal types include
  `ice-candidate`; the frontend sends `ice`. README's "verified" section
  does not mention that the live sign path and dashboard start flow were
  never exercised end to end. `backend/README.md` and root `README.md`
  duplicate run instructions.
- **Frontend nits.** `next.config.js` has no security headers;
  `NEXT_PUBLIC_API_URL` defaults to localhost (breaks the deployed
  frontend silently); MediaPipe is loaded from a CDN at runtime with no
  integrity pin and no offline fallback; `CaptionBar`/`VideoTile` are
  fine; `window.prompt` for correction is crude but functional.
- **Redis / Postgres / migrations:** spec names Redis and Postgres with
  migrations. The app uses SQLite + `create_all` on import and no Redis.
  That is a reasonable MVP choice, but it is a documented deviation, not
  an implementation — add Alembic before any production schema change.

## What is good — do not redesign

- Demo honesty: confidence caps, `requires_confirm`, `[DEMO]` labelling
  and the README's implemented/simulated/pending split are correct and
  should stay.
- Auth primitives: bcrypt + JWT with expiry, `get_current_user`, guest
  mode, and the test suite's shape are sound; the fixes tighten them
  rather than replacing them.
- The modular layout (`routes/` / `services/` / `ml/`, frontend
  `lib/` split by concern) is the right shape; the problems are wiring
  and validation, not structure.

## Suggested order after P0

P1-4 (room validation) → P1-1 (glare) + P1-2 (TURN) → P1-5 (settings
sync) → P1-3 (speech restart) → P1-7 (participants) + P1-10 (end room)
→ P0-4 second half (remote TTS injection) → P1-6 (AI singletons +
integration tests) → P1-8 (httpOnly cookies) → P2 hygiene (gitignore,
Docker, Alembic, CI, frontend tests).
