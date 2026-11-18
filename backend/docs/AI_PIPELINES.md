# AI Pipelines — SignSpeak Telecom (backend)

Real models, no simulations. Every pipeline below runs locally (no API
keys) and answers **HTTP 503 with a machine-readable reason** when it
cannot do its job — it never invents a prediction, transcript, or audio
file.

## 1. Sign recognition — MediaPipe → temporal PyTorch SignLSTM

```
browser MediaPipe landmarks (21x3) ──► POST /api/ai/sign/predict
        or server-side: POST /api/ai/sign/extract (image → landmarks)
            │  SignLSTM (2-layer LSTM, 63-dim features, 6 classes)
            ▼  softmax confidence, thresholds 0.85 / 0.60
        prediction + requires_confirm
```

- **Model**: `app/ml/models/sign_lstm.py` (SignLSTM, input 63, hidden 128
  by default). Weights live at `SIGN_MODEL_PATH`
  (`app/ml/models/sign_lstm.pt` by default) and are saved as a checkpoint
  dict `{state_dict, hidden_dim, input_dim, num_classes}` so inference
  always builds the architecture the weights were trained with.
- **Features**: `sequence_to_features` — wrist-origin translation, scale
  by wrist→middle-MCP distance, pad/truncate to 30×63.
- **Default path is trained-model only.** With no weights (or no torch)
  `predict()` raises `ModelNotTrainedError` and the route answers
  `503 {mode: "model-not-trained", trained: false}`. The old demo pose
  heuristics run **only** with `demo=true` and are labelled
  `demo-rule-based`, confidence capped at 0.78.
- **Server-side extraction**: `vision_service` is a lazy singleton
  MediaPipe **Tasks API** `HandLandmarker` (mediapipe ≥ 1.0 removed
  `mp.solutions`). Model file `backend/models/hand_landmarker.task`
  (7.8 MB). `POST /api/ai/sign/extract` returns `{landmarks: 21x3 |
  null, found: bool}`; 503 when MediaPipe or the model file is missing.
  System libs needed: `libEGL.so.1`, `libGLESv2.so.2` (Ubuntu:
  `libegl1`, `libgles2`).
- **Training data**: `POST /api/ai/sign/collect {label, sequence}`
  stores real user-recorded sequences (5–30 frames of 21×3 floats,
  label must be in the vocabulary, cap 200/label, oldest dropped) as
  JSON under `SIGN_SAMPLES_DIR` (`backend/data/sign_samples/<label>/`).
  No synthetic data in production code. `GET /api/ai/sign/samples`
  returns counts per label.
- **Training**: `POST /api/ai/sign/train {epochs, hidden_dim}` starts a
  background job (worker thread, request returns 202 immediately).
  `GET /api/ai/sign/train/{job_id}` → `{status: queued|running|done|
  failed, progress, epochs_done, val_accuracy, message}`. A second
  train while one runs → **409**. On success the weights are saved
  atomically and the inference cache is cleared, so the new model is
  live immediately. Train/val split is stratified (~20% held out,
  `val_accuracy` is `null` when nothing can be held out).
- **Async**: inference runs via `asyncio.to_thread` (`predict_async`);
  the model is a thread-locked lazy singleton (loaded once, never at
  import/startup). Duplicate suppression is per-caller (2 s window).

## 2. Whisper ASR → live captions

```
audio bytes (webm/opus or wav) ──► faster-whisper tiny, int8, CPU
        │  POST /api/ai/transcribe/audio   or   WS /ws/captions
        ▼
{text, language, engine: "whisper-tiny",
 confidence: 0–1 from avg_logprob, segments: [{start,end,text,confidence}],
 is_final: true}
```

- **Model**: faster-whisper `tiny`, int8, CPU, lazy singleton behind an
  asyncio lock (concurrent transcriptions serialise through the one
  instance). Model dir `backend/models/whisper-tiny/` (75 MB).
  **Pin**: `av==13.1.0` — faster-whisper 1.2.1 passes `metadata_errors`
  to `av.open`, which av ≥ 15 removed; av 14 has no cp312 wheel.
- **Confidence**: `exp(avg_logprob)` clamped to [0, 1] (segment-level
  and duration-weighted overall). Empty / no-speech audio returns
  `text ""` with low confidence — never invented words.
- **503** when the model cannot load (not downloaded / import fails).
- **Honesty rule**: `POST /api/ai/transcribe {"text": ...}` accepts
  client-supplied text (e.g. browser Web Speech API) and returns
  `engine: "client-text-passthrough"` with a note — it is **not**
  labelled Whisper.
- **WebSocket `/ws/captions?token=<jwt>`** (same JWT + 64 KB-per-message
  cap as `/ws/call`): binary audio chunks are accumulated; every ~2 s
  (or on `{"type":"flush"}`) the buffer is transcribed in a worker
  thread and the new text is sent as
  `{type:"caption", text, confidence, is_final:true, engine:"whisper-tiny"}`.
  `{"type":"text", ...}` client transcriptions are echoed back labelled
  `client-text-passthrough`. Failures arrive as `{type:"error", message}`
  — never fake captions. `{"type":"reset"}` clears the buffer.

## 3. Piper TTS + LiveKit voice agent

```
text (≤500 chars) ──► Piper en_US-lessac-medium (22050 Hz WAV)
        │  POST /api/ai/tts  →  /api/ai/tts/audio/<id>
        ▼  voice agent: resample → 48 kHz PCM → 10 ms LiveKit frames
LiveKit room participant "signspeak-voice" ("SignSpeak Voice")
```

- **TTS**: `tts_service` is a lazy Piper singleton (threadpool
  synthesis). `POST /api/ai/tts` writes a real WAV under
  `backend/data/tts/` and returns `{text, audio_url,
  engine:"piper-en_US-lessac-medium", voice, duration_s}`.
  `GET /api/ai/tts/audio/<id>` serves the WAV;
  `GET /api/ai/tts/voices` lists the Piper voices **actually installed**
  on disk. 503 when Piper or the voice is missing. Voice files:
  `backend/models/piper/en_US-lessac-medium.onnx[.json]` (63 MB).
- **Voice agent** (`app/services/voice_agent.py`, `livekit` package —
  note: the PyPI name is `livekit`, **not** `livekit-rtc`, which does
  not exist): one agent per room, max. Each agent owns a daemon thread
  with its own asyncio event loop (never the FastAPI loop). It mints a
  token via `livekit_service`, joins as identity `signspeak-voice`,
  publishes a real audio track, and speaks queued text serially.
  Piper's 22050 Hz output is resampled to 48 kHz with numpy linear
  interpolation (`np.interp` over the sample index — a real resample,
  not a relabel) and streamed as 10 ms frames (480 samples), paced at
  real time.
- **Endpoints**: `POST /api/ai/rooms/{id}/voice/start` (creator or
  participant; 409 if already running), `POST .../voice/speak {text}`
  (queues; returns `{spoken:true, duration_s}` or 503; auto-starts the
  agent if the room exists and the caller is a participant),
  `POST .../voice/stop`, `GET .../voice/status`. The agent leaves
  cleanly on room end — `POST /rooms/{id}/end` stops it best-effort.
- **Status**: `GET /api/ai/status` summarises all four pipelines
  honestly (sign mode/trained/torch/MediaPipe/samples, Whisper
  availability/model, TTS availability/voices, voice-agent rooms).

## Async / queues / memory

- Heavy work never blocks the FastAPI event loop: torch inference and
  Whisper transcription run in a threadpool (`asyncio.to_thread`),
  Piper synthesis in a threadpool, training in a worker thread,
  the voice agent in its own thread + event loop with an `asyncio.Queue`
  for serialised speech.
- Models are lazy singletons (load on first use, never at import).
  This machine has ~1 GB free RAM: torch CPU only, faster-whisper
  int8 tiny, Piper `en_US-lessac-medium`. Tests load one model at a
  time and release it (`release_model()` / `release_voices()` /
  `vision_service.release()` / `clear_model_cache()`).

## Error / 503 contract

| Situation | Response |
|---|---|
| No trained sign weights / torch missing | `503 {mode:"model-not-trained", trained:false}` |
| Whisper model not downloaded / cannot load | `503` with the reason (path + download command) |
| Piper / voice not installed | `503` with the reason |
| LiveKit unconfigured, or agent cannot join | `503` with the reason |
| MediaPipe / hand-landmarker model missing | `503` with the reason |
| Training already running | `409` |
| Text > 500 chars | `422` (schema) / `400` |
| Audio/image > 10 MB | `413` |

## What is REAL now vs what still needs what

| Component | Status |
|---|---|
| SignLSTM inference (trained path) | REAL — but **no trained weights ship with the repo**. First real weights need recorded samples (`POST /api/ai/sign/collect`, ≥2 labels) then `POST /api/ai/sign/train`. Demo heuristics exist only behind `demo=true`. |
| MediaPipe server-side extraction | REAL — `hand_landmarker.task` downloaded (7.8 MB). Needs `libEGL.so.1` + `libGLESv2.so.2` on the host (Ubuntu: `libegl1`, `libgles2`). |
| Whisper ASR + live captions | REAL — tiny int8 model downloaded (75 MB). |
| Piper TTS | REAL — `en_US-lessac-medium` downloaded (63 MB). |
| Voice agent (join + speak) | REAL code; **join/speak NOT verified end-to-end in this sandbox** (see below). Audio pipeline (synthesis → 48 kHz resample → 10 ms frames) is unit-tested for real. |
| LiveKit Cloud (production) | Still needs real credentials — dev keys `devkey`/`secret` are local-only (`backend/livekit-dev.md`). |

## Sandbox limitations (this environment, not the app)

- **livekit-rtc connect never completes here**: the server log shows
  the agent's join sessions starting, but the Rust client's
  `room.connect()` retries the signal connection and times out (60 s
  tested, with/without proxy env vars, `localhost` and `127.0.0.1`).
  The browser JS client connected to the same server fine. The voice
  agent reports this honestly as 503; `test_voice_agent_lifecycle`
  skips in this sandbox with the explanation. Verify join/speak/stop
  on a normal machine/network.
- **Media (remote video/audio) in LiveKit calls remains unverified
  here**: the sandbox blocks UDP sockets entirely and intercepts
  non-loopback TCP through an egress proxy, so ICE never completes
  (`backend/livekit-dev.md` §7).
- **Install notes**: torch needs the CPU wheel
  (`--index-url https://download.pytorch.org/whl/cpu`) and
  `TMPDIR=$HOME/.cache/pip-tmp` — pip's default temp dir is the 512 MB
  `/tmp` tmpfs and the ~800 MB wheel does not fit there.
