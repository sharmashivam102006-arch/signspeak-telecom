# SignSpeak Telecom — Frontend

Next.js 14 (App Router) + React 18 + TypeScript + Tailwind CSS frontend for SignSpeak Telecom.

## Run

```bash
cp .env.example .env.local   # API_URL defaults to http://localhost:8000
npm install
npm run dev                  # http://localhost:3000
npm run build                # production build
```

Backend is assumed at `NEXT_PUBLIC_API_URL` (default `http://localhost:8000`) with
the legacy caption/sign relay at `NEXT_PUBLIC_WS_URL/ws/call/{roomId}`.

## LiveKit calling (`/room/[id]`, `lib/livekit.ts`)

The call itself is **LiveKit** (`livekit-client` 2.x) — there is no P2P
fallback for media. Flow per call:

1. Validate the room (`GET /api/rooms/{id}` — 404/410 shows "Room not
   found" / "This room has ended" with a back-to-dashboard button).
2. `POST /api/rooms/{id}/join` (server presence).
3. `GET /api/rooms/{id}/token` → `{token, url, mode:"livekit"}`. The
   LiveKit URL comes from this response, never from env/hardcoding
   (`NEXT_PUBLIC_LIVEKIT_URL` in `.env.example` is informational only).
   If the backend returns 503 or `mode` is not `"livekit"` (or token/url
   is missing), the page shows the backend's error honestly and does not
   start a call.
4. `LiveKitCall.connect(url, token)`: `Room({adaptiveStream, dynacast})`,
   then `createLocalTracks` + publish. Remote participants arrive via
   `TrackSubscribed`/`ParticipantConnected`; their video/audio tracks are
   attached to `<video>`/`<audio>` elements with `track.attach()`.
   Remote audio elements are **not** muted; the local preview **is**
   muted (no feedback).

Controls — all drive LiveKit, not local state only:

- **Mic mute/unmute** → `localParticipant.setMicrophoneEnabled(bool)`; the
  published mic track is enabled/disabled in LiveKit and the remote side
  sees the mic icon change.
- **Camera on/off** → `localParticipant.setCameraEnabled(bool)`; the
  published camera track is enabled/disabled, and the tile shows
  "Camera off" (re-enabling re-acquires the camera).
- **End call** → `disconnect()` (unpublish + stop local tracks +
  `room.disconnect()`), `POST /api/rooms/{id}/leave`, navigate to
  `/dashboard`. **End room for everyone** (room creator only) →
  `POST /api/rooms/{id}/end`, then the same disconnect.
- **Participant panel** — live from LiveKit events
  (`ParticipantConnected/Disconnected`, `TrackMuted/Unmuted`,
  `ActiveSpeakersChanged`): name, mic/cam state, speaking indicator, count.

Camera/mic denied: `createLocalTracks` is retried per-device, so one
denial does not kill the other. You still join the room (visible
"camera unavailable" / "microphone unavailable" state, tile shows
"Camera off" / "Waiting for video…") and can re-enable the camera later
with the camera control, which re-acquires the device. Reconnecting is
handled by livekit-client; the status badge shows
Connecting… / Connected / Reconnecting… / Failed.

The old `lib/webrtc.ts` P2P client is legacy and unused by the call page.

## Pages

- `/` — landing page (hero, call preview, both AI pipelines, features, tech, vision)
- `/login`, `/signup` — JWT auth against `/api/auth/*`, plus guest/demo login
- `/dashboard` — protected; start/join calls, recent calls & transcripts (local), profile, accessibility settings, AI service status
- `/room/[id]` — the video call: LiveKit video/audio, participant list, live captions, sign recognition, TTS, transcript, confidence & correction

## AI pipelines (real backends, honest states)

All three pipelines call the backend contract in `lib/ai.ts` (`GET /api/ai/status`
is fetched once on entering a room and rendered as the **AI status row**:
Sign model Trained/Not trained/Unknown · Whisper Available/Unavailable ·
TTS engine · Voice agent On/Off/Unknown). When the backend does not expose an
endpoint yet (HTTP 404), the UI says so instead of inventing a state — the
current backend predates the new AI endpoints, so the status row reports
"unavailable" until it is deployed.

- **Sign recognition** (`lib/sign.ts` + `components/SignTrainer.tsx`):
  landmarks come from **MediaPipe HandLandmarker (`@mediapipe/tasks-vision`)**,
  loaded from the npm package with its wasm + `hand_landmarker.task` model
  bundled locally under `public/mediapipe/` — no CDN. (The legacy
  `@mediapipe/hands` `hands.js` CDN loader was tested in a real browser
  session and **does not load here** — the headless browser cannot reach
  the public internet — so it is kept only as a last-resort fallback; the
  tasks-vision loader is primary.) Landmarks are buffered into 30-frame
  sequences and sent to `POST /api/ai/sign/predict` at most once per
  ~800 ms, only when a hand is present and the sequence moved
  (`sequenceMoved`). A **503 model-not-trained** (or a demo-mode answer to
  a plain live call) is treated as "model not trained": no prediction is
  shown, the live loop stops, and a **"Sign model not trained yet"** card
  appears with per-sign sample counts and a **Train signs** button.
  Predictions carry a mode badge — **Trained model** vs **Demo** — plus
  confidence %. Thresholds from settings (default auto-accept ≥85%,
  confirm 60–85%, repeat <60%) drive the Accept / Correct / Repeat /
  Ignore card; Correct opens a vocabulary picker and `POST`s the corrected
  sequence to `/api/ai/sign/collect` as a training sample.
- **Training** (`components/SignTrainer.tsx`): per-sign sample counts from
  `GET /api/ai/sign/samples` (target 10/sign), a Record button capturing a
  2-second landmark sequence from the live camera (hand must be visible
  for ≥70% of the 2 s or the sample is discarded with a message), and a
  **Train model** button enabled when every sign has ≥5 samples. It
  `POST`s `/api/ai/sign/train` and polls the job endpoint, showing
  progress/epochs/val accuracy; on done it refreshes `/api/ai/status` and
  reports "Model trained — val accuracy X% — live recognition is now on".
  Backend errors are shown verbatim, never replaced with fake success.
- **Speech → captions** (`lib/captions.ts`): microphone audio is captured
  with `MediaRecorder` (webm/opus when supported) in ~2 s chunks and sent
  as binary over **WS `/ws/captions?token=…`**; interim partials render
  grey, final captions show confidence % and engine **"Whisper tiny"**.
  If the socket or Whisper is unavailable, the UI says so and falls back
  **only** to the browser Web Speech API, labelled **"Browser captions
  (not Whisper)"** — browser captions are never labelled Whisper. Captions
  are local-only: they are transcribed from your microphone and shown only
  to you, not sent to the remote participant.
- **Sign → speech / voice agent** (`lib/ai.ts`): an accepted sign (or text
  typed in the **"Say this"** box) triggers `POST
  /api/ai/rooms/{room_id}/voice/speak`; the UI shows **"Speaking…"** while
  it runs and the backend's error verbatim on failure. The voice-agent
  status indicator (On/Off/Unknown) comes from `GET
  /api/ai/rooms/{room_id}/voice/status`, with Start/Stop buttons. The
  remote participant hears the agent's LiveKit audio track — **browser
  `speechSynthesis` is never the remote-audio path**; it remains only as
  a local preview, clearly labelled **"Local preview only"** and gated by
  the existing mute setting. Demo-sign results are also local-preview
  only and are never sent to the voice agent.

### How to train the sign model from the UI

1. Join a room and open **Train signs** (AI communication panel).
2. For each of the 6 signs, hold the pose in front of your camera and
   press **Record** — 10 samples per sign is the target (training
   unlocks at 5 per sign).
3. Press **Train model** and wait for the job to finish (progress,
   epochs and val accuracy are shown live).
4. The status row flips to "Sign model: Trained" and live recognition
   starts; the "not trained" card disappears.

## Real vs demo — stated honestly

- Real: LiveKit video/audio calling (camera/mic publish, remote tracks, mute/camera controls, participant presence, leave/end-room), JWT auth, room creation/joining, transcript export, settings; MediaPipe HandLandmarker landmark extraction from the live camera (bundled locally); the full AI client (`lib/ai.ts`, `lib/captions.ts`) against the backend contract above, with honest 404/503 degradation when an endpoint is not deployed yet.
- Demo (labelled in the UI): the six demo-sign buttons (`Hello, Yes, No, Help, Thank You, Emergency`) call `POST /api/ai/sign/predict` with `demo=true` on the current landmark sequence and show the result with a **DEMO** badge and a local preview only. They are a test aid, never presented as model predictions, and need a hand in view.
- Pending on the backend (not yet deployed at the time of writing): the new AI endpoints — `/api/ai/status`, `/api/ai/sign/status|samples|collect|train|extract`, `/ws/captions`, `/api/ai/rooms/{id}/voice/*` — and the trained PyTorch sign model behind `POST /api/ai/sign/predict` (the current backend answers with demo heuristics only, which the frontend treats as "not trained"). Whisper is installed on the backend (`"whisper":"installed"` in `/health`) but the `/ws/captions` socket does not exist yet, so captions currently fall back to the browser Web Speech API.
- Sandbox limitation: the sandbox blocks UDP, so ICE never completes and `Room.connect()` rejects after ~15 s with "could not establish pc connection" — the room page surfaces that error honestly. Media (and therefore the in-room MediaPipe/Whisper flows, which need the live camera/mic stream) must be verified on a normal machine/network; see `backend/livekit-dev.md` §7.
