# LiveKit — local dev setup

SignSpeak's media plane is **real LiveKit** (no P2P fallback for calls).
The backend issues real LiveKit JWTs and manages rooms through the
LiveKit RoomService; the frontend joins with `livekit-client`.

## 1. Run the LiveKit server (dev mode)

A self-hosted binary is bundled at `~/workspace/signspeak-telecom/livekit/livekit-server`
(v1.13.11), with a config file at `livekit/livekit.yaml`. From the repo root:

```bash
cd ~/workspace/signspeak-telecom/livekit
./livekit-server --config livekit.yaml
```

The config starts the server on **ws://localhost:7880** with the dev
credentials (API key `devkey`, API secret `secret`), binds the signalling
and RTC ports to loopback, and advertises `node_ip: 127.0.0.1` so browsers
on the same machine get loopback ICE candidates. (`--dev` alone advertises
the machine's external IP, which fails in sandboxed environments.)
Leave it running in its own terminal. Stop with Ctrl-C.

> **Sandbox limitation (verified 2026-11-17):** this VM's sandbox blocks
> UDP socket creation entirely (seccomp) and intercepts non-loopback TCP
> through an egress proxy, so WebRTC media (ICE) cannot complete here —
> two browsers join the room and connect to LiveKit signalling, but the
> ICE transport never establishes and the call shows "Failed". The app
> code is not the cause: the same setup works on a normal machine/network
> where UDP is available. To verify media end-to-end, run the stack on a
> normal machine and join from two browsers (see §7).

## 2. Run the backend

```bash
cd ~/workspace/signspeak-telecom/backend
source .venv/bin/activate   # livekit-api is a required dep (requirements.txt)
cp .env.example .env        # dev defaults already point at ws://localhost:7880
uvicorn app.main:app --reload --port 8000
```

`GET /health` should report `"livekit_mode": "livekit"`. If it says
`"unconfigured"`, the LiveKit env vars are unset — the token endpoint
will answer 503 rather than hand out a fake token.

## 3. How the backend connects

| Piece | Where |
|---|---|
| Token (room = room code, identity = user id, `room_join`, creator gets `room_admin`) | `GET /api/rooms/{id}/token` → `app/services/livekit_service.py:generate_token` |
| Participant roster (join/leave/end) | `app/routes/rooms.py` + `RoomParticipant` table |
| RoomService: list participants, delete room | `app/services/livekit_service.py` (`livekit.api.LiveKitAPI`, async SDK wrapped with `asyncio.run`) |
| Webhook receiver | `POST /api/livekit/webhook` (`app/routes/livekit_webhook.py`) — verified with the SDK `WebhookReceiver` |

Room lifecycle: create room → `POST /join` (records participant) →
`GET /token` (join LiveKit) → `POST /leave` on hang-up →
`POST /end` (creator only; deletes the LiveKit room, best-effort).

## 4. Webhook URL

Point LiveKit at the backend webhook so the roster mirrors real room
state even if a client disconnects without calling `/leave`:

- Local dev server: start it with webhooks enabled, e.g. in
  `livekit.yaml`:
  ```yaml
  webhook:
    urls: ["http://localhost:8000/api/livekit/webhook"]
  ```
  then `./livekit-server --dev --config livekit.yaml`
  (or set `LIVEKIT_WEBHOOK_URLS` per the LiveKit docs for your setup).
- LiveKit Cloud: add `http://<your-backend-host>/api/livekit/webhook`
  in the project dashboard under Webhooks.

Verification uses the API key/secret (or `LIVEKIT_WEBHOOK_SECRET` when
set). Unverifiable or malformed events are logged and answered 200
`{received: false}` — never 500, so LiveKit does not retry-storm.

## 5. Production (LiveKit Cloud)

1. Create a project at https://cloud.livekit.io — copy its URL, API key
   and API secret.
2. Set in the backend environment (never commit them):
   ```
   LIVEKIT_URL=wss://<your-project>.livekit.cloud
   LIVEKIT_API_KEY=<real key>
   LIVEKIT_API_SECRET=<real secret>
   ENVIRONMENT=production
   ```
   The dev defaults (`devkey` / `secret`) must not be used in production.
3. Configure the webhook URL in the Cloud dashboard (see §4).
4. The frontend needs `NEXT_PUBLIC_LIVEKIT_URL` pointing at the same
   Cloud URL (it joins with the token from `/api/rooms/{id}/token`).

## 6. Optional TURN relay (UDP-blocked environments only)

`frontend/lib/livekit.ts` accepts an optional TURN server from env
(`NEXT_PUBLIC_TURN_URL`, `NEXT_PUBLIC_TURN_USERNAME`,
`NEXT_PUBLIC_TURN_CREDENTIAL`, e.g.
`turn:127.0.0.1:3478?transport=tcp`). When set, the client passes it as
`rtcConfig.iceServers` to `room.connect()`; when unset, LiveKit's own ICE
servers from the JoinResponse are used. This is an escape hatch for
networks where direct ICE cannot work — it does not replace LiveKit, and
production should use LiveKit Cloud's built-in TURN instead.

## 7. Two-browser-session test (what was verified 2026-11-17)

Playwright (headless Chromium + Firefox) with fake camera/mic devices:

- ✅ Guest login → Start New Call → room created (`/room/<CODE>`)
- ✅ Second session joins with the code → both land on the same room URL
- ✅ `GET /api/rooms/{code}/participants` lists **both** participants, active
- ✅ Both browsers connect to LiveKit signalling (RTC sessions start)
- ✅ Mute/camera buttons call the real LiveKit local-participant methods
  and flip state; End call → dashboard + `POST /leave` 200
- ❌ **Not verified in this sandbox:** remote video/audio rendering and
  mute/camera propagation to the other participant — ICE never completes
  here (see the sandbox limitation in §1). Verify on a normal machine:
  join the same room from two real browsers, check each sees the other's
  video, mute/unmute and camera on/off reflect on the remote tile, and
  ending the call removes the participant for the other side.

## 8. Quick smoke test (no frontend needed)

```bash
# 1. start livekit-server --dev and the backend (above)
# 2. register + create a room, then fetch a token:
TOKEN=$(curl -s -X POST localhost:8000/api/auth/guest -H 'Content-Type: application/json' -d '{}' | python3 -c 'import sys,json;print(json.load(sys.stdin)["access_token"])')
ROOM=$(curl -s -X POST localhost:8000/api/rooms -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' -d '{"name":"Smoke"}' | python3 -c 'import sys,json;print(json.load(sys.stdin)["id"])')
curl -s localhost:8000/api/rooms/$ROOM/token -H "Authorization: Bearer $TOKEN"
# -> {"room_id": "...", "code": "XXXXXX", "token": "eyJ...", "url": "ws://localhost:7880", "mode": "livekit", "note": ""}
```
