# SignSpeak Telecom — API Reference

Rewritten 2026-11-18 from the live OpenAPI spec (`GET /openapi.json` on the
running backend) and `backend/app/main.py`. If this file and `/openapi.json`
disagree, `/openapi.json` wins — regenerate this table from it.

Base URL (local): `http://localhost:8000`. Interactive docs: `GET /docs`.
Auth: `Authorization: Bearer <JWT>` unless noted. Guest tokens work the same
way. "Auth" below is what the live spec declares; endpoints marked **open**
that do heavy AI work are being restricted under Priority 1 (security) —
re-check the spec after that lands.

## Health & root
| Method & path | Auth | Purpose / response |
|---|---|---|
| `GET /` | open | `{service, docs, health}` |
| `GET /health` | open | `{status:"ok", service, livekit_mode:"livekit"\|"unconfigured", sign_mode:"trained-model"\|"model-not-trained", whisper:"installed"\|"not-installed", tts:"piper"\|"not-installed"}` |

## Auth — `/api/auth`
| Method & path | Auth | Purpose |
|---|---|---|
| `POST /api/auth/register` | open | `{name, email, password}` → 201 `{access_token, token_type:"bearer", user}`. Passwords bcrypt-hashed. Duplicate email → 400. |
| `POST /api/auth/login` | open | `{email, password}` → `{access_token, ...}`. Wrong password → 401. |
| `POST /api/auth/guest` | open | `{}` → 201, creates a guest user + JWT. No registration needed. |
| `GET /api/auth/me` | auth | Current user. |

## Rooms — `/api/rooms`
| Method & path | Auth | Purpose |
|---|---|---|
| `GET /api/rooms` | auth | Rooms created by / joined by the user. |
| `POST /api/rooms` | auth | `{title?}` → 201 room `{id, code (6 chars), title, invite_link: <frontend>/room/<code>, created_by, created_at}`. |
| `GET /api/rooms/{room_id}` | open | Room by UUID or 6-char code. Unknown → 404. (Auth on this lookup is part of Priority 1.) |
| `POST /api/rooms/{room_id}/join` | auth | Adds the caller as a participant; returns the room. |
| `POST /api/rooms/{room_id}/leave` | auth | Removes the caller from the room. |
| `GET /api/rooms/{room_id}/participants` | auth | Active participants in the room. |
| `POST /api/rooms/{room_id}/end` | auth | Ends the room (creator only); ended rooms return 410 afterwards. |
| `GET /api/rooms/{room_id}/token` | auth | Real LiveKit JWT for this room (room = room code, identity = user id, creator gets `room_admin`). 503 when LiveKit is unconfigured — there is no P2P fallback. |
| `POST /api/rooms/{room_id}/signal` | auth | Stores a signalling message `{type, payload}` for the room (REST fallback for the WS channel). |
| `GET /api/rooms/{room_id}/signal` | auth | Reads pending signalling messages for the room. |

## Transcripts — `/api/rooms/{room_id}/transcripts`
| Method & path | Auth | Purpose |
|---|---|---|
| `GET /api/rooms/{room_id}/transcripts` | auth | Entries `{id, room_id, direction:"sign-to-speech"\|"speech-to-text", text, confidence?, source, created_at}`, creator-only. |
| `POST /api/rooms/{room_id}/transcripts` | auth | `{direction, text, confidence?, consent:true}` → 201. `consent:false` → 400 (nothing is stored without consent). |
| `GET /api/rooms/{room_id}/transcripts/export` | auth | Plain-text transcript download. |

## Sign recognition — `/api/ai/sign`
| Method & path | Auth | Purpose |
|---|---|---|
| `GET /api/ai/sign/vocabulary` | open | `{vocabulary:["Hello","Yes","No","Help","Thank You","Emergency"], count:6}`. |
| `GET /api/ai/sign/status` | open | Model state: `{mode, trained, torch_available, mediapipe_available, model_path, samples_per_sign, vocabulary}`. `trained:false` until real weights exist. |
| `POST /api/ai/sign/predict` | auth | `{sequence: [[[x,y,z] x21] x N], demo?, force?}` → `{prediction, confidence, mode, requires_confirm, thresholds:{high:0.85, low:0.60}}`. Modes: `"trained-model"` (real softmax confidence), `"demo-rule-based"` (only when `demo:true`; fixed heuristic confidences, capped 0.78), `"model-not-trained"` (HTTP 503 — no weights yet), `"no-hand"`. Same prediction within 2 s is suppressed (`duplicate:true`). |
| `POST /api/ai/sign/extract` | open | Multipart image → MediaPipe hand extraction `{found, landmarks: [[x,y,z] x21]}`. Hand-only, 1 hand max. |
| `POST /api/ai/sign/collect` | auth | Stores one recorded sequence: `{label, sequence}` → `{label, count}`. Samples are saved as JSON under `SIGN_SAMPLES_DIR/<label>/<uuid>.json`. |
| `GET /api/ai/sign/samples` | open | Per-sign recorded sample counts `{samples_per_sign, total}`. |
| `POST /api/ai/sign/train` | open | Queues a training job → `{job_id, status}` (409 when one is already running). Trains the SignLSTM on recorded samples only — never invents data. (Auth/authorization on this endpoint is part of Priority 1.) |
| `GET /api/ai/sign/train/{job_id}` | open | Poll a training job: `{job_id, status:"queued"\|"running"\|"done"\|"failed", progress, epochs_done, val_accuracy, message}`. `val_accuracy` is null when no label had enough samples to hold any out. |

## Speech — `/api/ai/transcribe`
| Method & path | Auth | Purpose |
|---|---|---|
| `POST /api/ai/transcribe` | open | Accepts client-transcribed text (e.g. browser Web Speech API) into the caption pipeline. The response labels it `source:"client-stt"` — it was not transcribed by Whisper. |
| `POST /api/ai/transcribe/audio` | open | Multipart audio → real faster-whisper transcription `{text, language, segments, confidence, source:"whisper"}`. 503 while faster-whisper is not installed. |

## Text-to-speech — `/api/ai/tts`
| Method & path | Auth | Purpose |
|---|---|---|
| `POST /api/ai/tts` | open | `{text, voice?}` → Piper synthesis `{audio_url:"/api/ai/tts/audio/<id>", engine:"piper", duration_s}`. Text is capped at 500 chars. 503 while Piper is not installed. |
| `GET /api/ai/tts/voices` | open | Installed Piper voices `{voices:[...], engine}`. |
| `GET /api/ai/tts/audio/{audio_id}` | open | Serves a generated WAV by its hex id. |

## Voice agent — `/api/ai/rooms/{room_id}/voice`
| Method & path | Auth | Purpose |
|---|---|---|
| `POST /api/ai/rooms/{room_id}/voice/start` | auth | Starts the "SignSpeak Voice" LiveKit participant in the room. |
| `POST /api/ai/rooms/{room_id}/voice/speak` | auth | `{text}` → the agent synthesises it with Piper and publishes the audio into the room. Returns `{spoken, duration_s}`. 503 when the agent cannot join LiveKit. |
| `POST /api/ai/rooms/{room_id}/voice/stop` | auth | Stops the voice agent in the room. |
| `GET /api/ai/rooms/{room_id}/voice/status` | auth | `{active, rooms_active}` — whether the agent is running in this room. |

## Users — `/api/users`
| Method & path | Auth | Purpose |
|---|---|---|
| `GET /api/users/me/preferences` | auth | Caption size, high contrast, TTS mute/volume/voice, sign thresholds. |
| `PUT /api/users/me/preferences` | auth | Update the same preferences. |

## AI status — `/api/ai`
| Method & path | Auth | Purpose |
|---|---|---|
| `GET /api/ai/status` | open | Whole-pipeline status `{sign, whisper, tts, voice_agent}` (same shapes as the per-pipeline endpoints above). |

## LiveKit webhook
| Method & path | Auth | Purpose |
|---|---|---|
| `POST /api/livekit/webhook` | open (signature) | LiveKit server events (`room_started`, `room_finished`, `participant_joined`, …). Verified against `LIVEKIT_WEBHOOK_SECRET` (falls back to `LIVEKIT_API_SECRET`). Configure the webhook URL in your LiveKit server config; unverified events are rejected when a secret is configured. |

## WebSockets (not in the OpenAPI paths — see `backend/app/main.py`)
| Socket | Auth | Purpose |
|---|---|---|
| `WS /ws/call/{room_id}?token=<jwt>` | JWT + room membership | Relays `caption` / `sign` / `tts` / `transcript` messages between peers in the same room. Requires a valid JWT and an existing active room (close codes 4401/4404 otherwise). 64 KB message cap. Reconnect: re-open the socket and re-send `join`. Media itself is NOT carried here — media is LiveKit. |
| `WS /ws/captions?token=<jwt>` | JWT | Live Whisper captions: send ~2 s audio chunks (binary), receive `{type:"final", text, confidence, engine:"whisper-tiny"}` (and `partial` when the engine supports it). Captions are returned to the speaker's own socket; delivery to the remote participant goes through `/ws/call` (Priority 3). 64 KB message cap. |

## Errors
- `400` validation / `401` bad or missing token / `403` not a member or not the creator / `404` unknown room or endpoint / `409` training job already running / `410` room ended.
- `503` means a real dependency is unavailable, never a fake success: `model-not-trained` (no sign weights), LiveKit unconfigured, Whisper/Piper not installed, or the voice agent could not join the room. The response body always carries the reason in `detail`.
