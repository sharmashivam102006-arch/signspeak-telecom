# Architecture

## Layers

```
┌─ Accessibility UX ────────────────────────────────────────────────┐
│  caption scale · high contrast · reduced motion · focus order     │
├─ Delivery ────────────────────────────────────────────────────────┤
│  live caption rail (immediate)   speech synthesis (TTS track)     │
├─ Understanding ───────────────────────────────────────────────────┤
│  sign: segment → classify → calibrate → decide   [browser]        │
│  speech: VAD → ASR → partial/final                [browser|server]│
├─ Perception ──────────────────────────────────────────────────────┤
│  MediaPipe HandLandmarker + PoseLandmarker (+Face) [browser]      │
├─ Communication ───────────────────────────────────────────────────┤
│  LiveKit / WebRTC: audio, video, reliable data channel            │
├─ Control plane ───────────────────────────────────────────────────┤
│  FastAPI: identity, rooms, tokens, transcript, telemetry          │
│  Postgres (durable)   Redis (ephemeral session state)             │
└───────────────────────────────────────────────────────────────────┘
```

The boundary that matters most is between **Perception/Understanding** and
everything else. Both run entirely in the browser. No frame, no landmark and no
feature vector crosses the network. What travels is a word and a calibrated
number.

That is not only a privacy property. It is also why recognition survives a bad
connection: LiveKit may drop the *transmitted* video quality under congestion,
but the local camera track is untouched, and recognition reads the local track.
A server-side recogniser would degrade exactly when the user most needs it to
work.

---

## Why these choices

### LiveKit rather than raw WebRTC
Signalling, SFU, reconnection, simulcast and TURN are each a week of work and
none of them is the project. The token minted in `rooms.py` is
least-privilege — `room_create` and `room_admin` are deliberately absent, so a
leaked participant token cannot destroy the room.

### Captions on the data channel, presence on the WebSocket
The LiveKit data channel rides the same connection as the media, so a caption
cannot arrive before the video it belongs to, and it keeps working while the
API server restarts. The app WebSocket carries presence and clock sync, where
ordering against media is irrelevant.

Captions are published **reliable**, never lossy. Lossy mode is for data you
resend constantly; a caption is sent once and is one participant's only access
to what was said.

### Two consumers of the same events, two ordering policies
- **Live caption rail** renders on arrival. Briefly out of order is far less
  confusing than a second late.
- **Stored transcript** passes through a ~900ms reorder buffer keyed on
  utterance *start* time. A 1.4s sign that began before a 0.3s word belongs
  above it.

Client clocks have no shared origin, so `services/ordering.py` estimates a
per-participant offset from ping/pong round trips, NTP-style, keeping the
lowest-RTT sample.

### Redis is not the database
Room presence, socket state, rate limits, in-flight partials. Everything in
Redis can be lost without losing a conversation.

---

## Latency budget

Measured on the pipeline in this repo, 25fps, mid-range laptop, CPU WASM:

| stage | typical | notes |
| --- | --- | --- |
| MediaPipe landmarks | 8–15 ms/frame | overlapped with capture, not additive |
| feature encoding | < 0.5 ms | pure arithmetic |
| **end-of-sign detection** | **~280 ms** | `exit_frames=7` at 25fps |
| classifier (GRU, ONNX Web) | 2–5 ms | WebGPU ≈ 1 ms |
| decision + policy | < 0.1 ms | |
| TTS start | 50–150 ms | browser; cloud TTS 200–500 ms |
| WebRTC audio path | 100–200 ms | network dependent |
| **sign finished → heard** | **≈ 450–650 ms** | |

**Segmentation lag is ~60% of the controllable budget and inference is under
1%.** This inverts the usual instinct to optimise the model. The lever is
`exit_frames`, and shortening it starts splitting slow signs into two
detections — a real trade-off with a curve you can plot and report.

The principled fix is early-commit streaming classification: classify the
growing prefix every few frames, emit when calibrated confidence crosses
threshold, never wait for the sign to end. Cost: the GRU must become
**unidirectional**, so the bidirectional accuracy measured offline stops being
the accuracy users experience. Measure both before claiming either.

---

## Data flow, sign direction

```
camera (local track, full quality)
  └─ MediaPipe → {pose, leftHand, rightHand, face?}
      └─ encodeFrame()                      body-anchored, 184 floats
          └─ Segmenter.push()               hysteresis state machine
              └─ [segment: 8–90 frames]
                  └─ addVelocity + resample → 32 × 368
                      └─ ONNX GRU           → 7 logits
                          └─ ÷ temperature  → calibrated probs
                          └─ energy score   → OOD gate
                              └─ decide()   → speak | confirm | reject
                                  ├─ speak  → caption + TTS track + telemetry
                                  ├─ confirm→ signer sees a question, nothing spoken
                                  └─ reject → signer sees "didn't catch that"
```

Only the last stage produces network traffic, and only for `speak`. Rejections
are logged locally as telemetry (labels and scores, never landmarks) because
they are the most useful evaluation data the system produces.

---

## Data flow, speech direction

```
microphone
  └─ Web Speech API (default)               streaming, free, no infra
     or → PCM16 @16kHz → WS → EnergyVAD → faster-whisper (fallback)
         └─ partial hypotheses  → caption rail only, never persisted
         └─ final hypothesis    → caption rail + transcript (with consent)
```

Whisper is not a streaming model. Every "streaming Whisper" re-runs it on a
growing buffer, so text is unstable — words change after appearing, which is
disorienting for someone who cannot hear the audio to check. Web Speech is the
default for that reason; Whisper exists for Firefox, for Hindi, and for
deployments where third-party audio processing is unacceptable.

**Privacy asymmetry, to state explicitly:** Chrome's Web Speech sends audio to
Google. The "nothing leaves the device" claim is true for the sign direction
and false for the speech direction. Do not let one claim cover both.

---

## Privacy and data minimisation

| data | where it lives | retention |
| --- | --- | --- |
| video / audio frames | the user's device and the WebRTC path | never stored |
| landmarks, features | browser memory only | discarded per frame |
| recognised label + confidence | LiveKit data channel, optionally Postgres | 30 days, opt-in |
| transcript text | Postgres, only with per-participant consent | 30 days, deletable |
| recognition telemetry | Postgres, labels and scores only | 30 days |

Consent is per person per call, timestamped. The host enabling transcripts is a
*request*; `join_room` turns it back off if the joining participant declines.
Either participant can delete the transcript at any time without explanation.

The schema has **no disability field**. The system needs to know which channel
to render; it does not need to know anything about the person. `Modality` is a
communication preference, and in the target sectors — healthcare, banking,
employment — storing disability status would be both unnecessary and regulated.

---

## Scaling shape

Media, control and AI scale on independent axes:

- **Media** — LiveKit SFU, scales on bandwidth. Add nodes.
- **Control** — FastAPI, stateless except for the in-process caption fan-out.
  `state.py` is explicitly documented as single-instance: two replicas behind a
  load balancer put two participants of one call on different processes and
  they stop seeing each other's captions. Either run one instance or enable the
  Redis pub/sub bridge. **Do not discover this during the demo.**
- **AI** — zero server cost. Recognition runs on the client, so per-call AI
  cost is zero and capacity is the number of browsers, not the number of GPUs.
  This is the strongest commercial argument in the whole design and it comes
  free from the privacy decision.

The one server-side AI cost is the Whisper fallback, which is CPU-bound and
guarded by a semaphore — two concurrent transcriptions on an int8 CPU model
each take twice as long rather than running in parallel, which turns "works
with one user" into "unusable with three" if left unbounded.
