# SignSpeak Telecom — First recording session plan

**Status checked 2026-11-18:** 0 sign samples, 0 signers
(`backend/data/sign_samples/` is empty; `GET /api/ai/sign/status`
reported all six signs at 0 on 2026-11-17). No trained weights exist —
`POST /api/ai/sign/predict` honestly answers 503 `model-not-trained`.
The collection + training + evaluation pipeline is complete and tested;
only real human recordings are missing. This plan is the next step.

## 0. Fix the admin gate first (blocking, 2 minutes)

Training is admin-only (`POST /api/ai/sign/train` → 403 otherwise), and
**no admin is configured right now**: `ADMIN_EMAILS` / `ADMIN_USER_IDS`
appear in `backend/app/config.py` only — they are in neither
`backend/.env` nor `backend/.env.example`, so the allowlist is empty
and nobody can train.

1. In `backend/.env`, add the email of the account you sign in with:
   `ADMIN_EMAILS=<your login email>`
   (or `ADMIN_USER_IDS=<your user id>` — email is simpler).
2. Restart the backend so the new env var loads.
3. Sanity check: a non-admin train attempt must 403; your train must
   return 202. If you get 403 as yourself, the email in `.env` does not
   match the account you signed in with.

## 1. Start the stack

From `~/workspace/signspeak-telecom/` (see `backend/livekit-dev.md` §1–2
and the implementation report §10):

```bash
# Terminal 1 — LiveKit (self-hosted v1.13.11)
cd livekit && ./livekit-server --config livekit.yaml   # ws://localhost:7880

# Terminal 2 — backend
cd backend && source .venv/bin/activate
uvicorn app.main:app --reload --port 8000
# GET /health should report "livekit_mode": "livekit"

# Terminal 3 — frontend (build first if needed)
cd frontend && npm run build && npx next start -p 3000
```

Note: the servers were **not running** when this plan was written
(2026-11-18 morning) — start all three before recording.

## 2. Who records — you need a second signer

- **Minimum for an honest model:** 2 signers × 6 signs × 5 samples.
- **Target:** 2–3 signers × 6 signs × 10 samples = 120 samples.
- Each signer needs **their own account** (signer id comes from the
  JWT; anonymous samples are not accepted). The second signer signs in
  on their own browser/device and records the same six signs.
- Why two matter: with ≥ 2 signers, training holds out one whole signer
  for validation (`split: "by_signer"`). With one signer only, the split
  falls back to `random` — the model has seen the same hands in
  training, so the metrics prove much less.
- **Decision needed before you start:** who is the second signer?

## 3. The session (Train Signs panel, in a call)

Vocabulary — Hello, Yes, No, Help, Thank You, Emergency. These are
generic introductory hand signs, **not certified Indian Sign Language**;
leave the language label as `generic` unless you can confirm a sign
is ISL (see `backend/docs/DATASET.md`).

Per signer, per sign:

1. Open a call, open the **Train Signs** panel.
2. Pick a sign, hold your hand in view, hit **Record “<sign>” (2s)**.
3. The panel discards the take if the hand was visible < 70% of the
   2 seconds or fewer than 5 frames were captured — re-record it.
4. Vary each take: different hand position, speed, background/lighting.
   Do not repeat the same frozen pose — variation is the data.
5. Repeat to 10 per sign (panel shows `n/10`; training unlocks at 5).

**Time:** pure recording is 2 min per signer at target
(60 takes × 2 s), ~4 min for two signers. With resetting between takes,
budget **15–25 minutes per signer**; the 5-per-sign minimum is about
10 minutes per signer. Do the minimum first if time is short — you can
add more samples and retrain later.

Progress check any time: `GET /api/ai/sign/status` reports
`samples_per_sign`, `samples_per_signer`, `total_signers`.

## 4. Train (admin only)

In the panel, **Train model** (or `POST /api/ai/sign/train`, defaults
epochs 30 / hidden 128; one job at a time, a second request gets 409).
Poll `GET /api/ai/sign/train/{job_id}`.

Accept the model only if the job result says:

- `split: "by_signer"` (or `"random"` if only one signer contributed),
- `val_available: true`,
- every sign has precision / recall / F1 with real support, plus a
  confusion matrix.

If a sign scores poorly, record more varied samples for that sign and
retrain — do not lower the 0.85 / 0.60 thresholds to hide it. On
success, weights land at `app/ml/models/sign_lstm.pt`,
`GET /api/ai/sign/status` reports `"trained": true`, and
`GET /health` reports `sign_mode: "trained-model"`.

## 5. Verify live

Perform each of the six signs in front of a camera and confirm
`POST /api/ai/sign/predict` returns `mode: "trained-model"` with real
softmax confidences (≥ 0.85 shown directly; 0.60–0.85 asks to confirm;
< 0.60 asks to repeat). Record the observed per-sign results in
`backend/docs/DATASET.md` before claiming recognition works.

## 6. The other half: real-network media test (same day, different machine)

Remote video/audio was **never verified** — this sandbox blocks WebRTC
ICE (UDP blocked, non-loopback TCP proxied), so two browsers join but
media never connects. Do this on a normal machine/network
(`backend/livekit-dev.md` §7):

1. Start the same stack on that machine; join the same room from two
   real browsers (two people, or two browsers on two devices).
2. Check each side sees the other's video (`videoWidth > 0` both sides).
3. Mute / unmute and camera on / off on one side — confirm it reflects
   on the remote tile.
4. End the call — confirm the other side sees the participant leave.
5. Voice agent: in the call, sign a recognised sign (needs §4–5 done)
   or use the speak control, and confirm the other participant hears
   the Piper voice. Also confirm the mic → Whisper → caption glue:
   speak, and the other side should see the live caption with your name.
6. If any of these fail on a normal network, that is a real app bug —
   note exactly which step, and it becomes the next fix, not an
   environment limit.

## What this does NOT need

- No new code, no rebuild, no dataset download — the pipeline, the
  panel, the evaluation and the docs already exist.
- No synthetic samples — never invent training data; fixtures live only
  in `backend/tests/`.
