"""API tests — run: python -m pytest  (from backend/) or python -m pytest tests/"""
import json
import os
import sys

os.environ["DATABASE_URL"] = "sqlite:///./test_signspeak.db"
os.environ["SECRET_KEY"] = "test-secret"
# LiveKit dev server credentials (see backend/livekit-dev.md). The token
# tests only need the SDK + these keys to verify the JWT; no LiveKit
# server needs to be running (RoomService calls are best-effort).
os.environ["LIVEKIT_URL"] = "ws://localhost:7880"
os.environ["LIVEKIT_API_KEY"] = "devkey"
os.environ["LIVEKIT_API_SECRET"] = "secret"
# Tests that need "no trained model" must not pick up the real weights
# shipped at app/ml/models/sign_lstm.pt; point the default model path at
# a file that does not exist. (test_shipped_sign_model loads the real one.)
os.environ["SIGN_MODEL_PATH"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "no_such_model.pt")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient

from app.database import Base, engine

Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)

from app.main import app  # noqa: E402

client = TestClient(app)


def _auth_headers():
    import uuid as _uuid
    email = f"test-{_uuid.uuid4().hex[:8]}@example.com"
    r = client.post("/api/auth/register", json={
        "email": email, "name": "Test User", "password": "password123"})
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}, email


def _open_palm():
    # 21 landmarks: open palm — all fingertips above (smaller y than) PIPs.
    lm = [[0.5, 0.9, 0.0]] * 21
    lm = [list(p) for p in lm]
    lm[9] = [0.5, 0.7, 0.0]  # middle MCP (scale anchor)
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        lm[pip] = [0.5, 0.6, 0.0]
        lm[tip] = [0.5, 0.4, 0.0]
    lm[3] = [0.5, 0.6, 0.0]; lm[4] = [0.5, 0.7, 0.0]  # thumb not up
    return lm


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_register_login_me():
    headers, email = _auth_headers()
    r = client.get("/api/auth/me", headers=headers)
    assert r.status_code == 200 and r.json()["email"] == email
    r = client.post("/api/auth/login", json={"email": email, "password": "password123"})
    assert r.status_code == 200 and "access_token" in r.json()
    r = client.post("/api/auth/login", json={"email": email, "password": "wrongpass"})
    assert r.status_code == 401


def test_guest():
    r = client.post("/api/auth/guest", json={"name": "Guest"})
    assert r.status_code == 201 and r.json()["user"]["is_guest"] is True


def test_create_room_and_join():
    headers, _ = _auth_headers()
    r = client.post("/api/rooms", json={"name": "Test Call"}, headers=headers)
    assert r.status_code == 201, r.text
    room = r.json()
    assert len(room["code"]) == 6 and room["invite_link"]
    r = client.get(f"/api/rooms/{room['code']}", headers=headers)
    assert r.status_code == 200
    r = client.post(f"/api/rooms/{room['code']}/join", json={}, headers=headers)
    assert r.status_code == 200


def test_join_invalid_room_404():
    headers, _ = _auth_headers()
    r = client.post("/api/rooms/XXXXXX/join", json={}, headers=headers)
    assert r.status_code == 404
    r = client.get("/api/rooms/XXXXXX", headers=headers)
    assert r.status_code == 404


def test_sign_vocabulary_and_demo_predict():
    r = client.get("/api/ai/sign/vocabulary")
    assert r.status_code == 200
    assert r.json()["vocabulary"] == ["Hello", "Yes", "No", "Help", "Thank You", "Emergency", "Please", "Sorry", "Good", "Goodbye"]
    # Default live path with no trained weights -> 503 model-not-trained
    # (never a silent heuristic presented as recognition).
    r = client.post("/api/ai/sign/predict", json={"landmarks": _open_palm(), "force": True})
    assert r.status_code == 503, r.text
    assert r.json()["detail"]["mode"] == "model-not-trained"
    assert r.json()["detail"]["trained"] is False
    # Explicit demo=true -> labelled demo heuristics.
    r = client.post("/api/ai/sign/predict",
                    json={"landmarks": _open_palm(), "force": True, "demo": True})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["mode"] == "demo-rule-based"
    assert data["prediction"] == "Hello"
    assert 0.55 <= data["confidence"] <= 0.85  # demo never auto-accepts
    assert data["requires_confirm"] is True
    # No hand
    r = client.post("/api/ai/sign/predict", json={"landmarks": []})
    assert r.status_code == 200 and r.json()["mode"] == "no-hand"


def test_tts_and_voices():
    from app.services import tts_service
    headers, _ = _auth_headers()  # POST /api/ai/tts requires authentication
    r = client.post("/api/ai/tts", json={"text": "Hello"}, headers=headers)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["text"] == "Hello"
    assert data["engine"] == "piper-en_US-lessac-medium"
    assert data["audio_url"] and data["audio_url"].startswith("/api/ai/tts/audio/")
    assert data["duration_s"] and data["duration_s"] > 0
    # The returned URL serves a real WAV file.
    audio_id = data["audio_url"].rsplit("/", 1)[-1]
    r = client.get(f"/api/ai/tts/audio/{audio_id}")
    assert r.status_code == 200 and r.headers["content-type"].startswith("audio/wav")
    assert r.content[:4] == b"RIFF" and len(r.content) > 1000
    r = client.get("/api/ai/tts/voices")
    assert r.status_code == 200 and r.json()["voices"]
    assert any(v["id"] == "en_US-lessac-medium" for v in r.json()["voices"])
    # Over the 500-char limit -> 422.
    r = client.post("/api/ai/tts", json={"text": "x" * 501}, headers=headers)
    assert r.status_code == 422
    tts_service.release_voices()  # free RAM before the Whisper tests


def test_transcribe_and_transcripts():
    headers, _ = _auth_headers()
    # Client-supplied text is labelled honestly, NOT as Whisper output.
    r = client.post("/api/ai/transcribe", json={"text": "Hello, can you hear me?"})
    assert r.status_code == 200 and r.json()["text"] == "Hello, can you hear me?"
    assert r.json()["engine"] == "client-text-passthrough"
    assert "not transcribed by Whisper" in r.json()["note"] or "client" in r.json()["note"]
    room = client.post("/api/rooms", json={"name": "T"}, headers=headers).json()
    r = client.post(f"/api/rooms/{room['id']}/transcripts",
                    json={"speaker_name": "A", "direction": "speech-to-text", "text": "Hi",
                          "consent": True},
                    headers=headers)
    assert r.status_code == 201
    r = client.get(f"/api/rooms/{room['id']}/transcripts", headers=headers)
    assert r.status_code == 200 and len(r.json()) == 1
    r = client.get(f"/api/rooms/{room['id']}/transcripts/export", headers=headers)
    assert r.status_code == 200 and "Hi" in r.text


def test_websocket_signalling():
    headers, _ = _auth_headers()
    room = client.post("/api/rooms", json={"name": "WS"}, headers=headers).json()
    token = headers["Authorization"].split()[1]
    with client.websocket_connect(f"/ws/call/{room['id']}?token={token}") as ws1, \
         client.websocket_connect(f"/ws/call/{room['id']}?token={token}") as ws2:
        ws1.send_text('{"type": "caption", "payload": {"text": "hello"}}')
        msg = ws2.receive_text()
        assert "hello" in msg


def test_invite_link_uses_room_route():
    headers, _ = _auth_headers()
    r = client.post("/api/rooms", json={"name": "Invite"}, headers=headers)
    assert r.status_code == 201
    assert "/room/" in r.json()["invite_link"]
    assert "/call/" not in r.json()["invite_link"]


def test_websocket_rejects_missing_token():
    from starlette.websockets import WebSocketDisconnect
    headers, _ = _auth_headers()
    room = client.post("/api/rooms", json={"name": "WSAuth"}, headers=headers).json()
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(f"/ws/call/{room['id']}"):
            pass


def test_websocket_rejects_nonexistent_room():
    from starlette.websockets import WebSocketDisconnect
    headers, _ = _auth_headers()
    token = headers["Authorization"].split()[1]
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(f"/ws/call/does-not-exist?token={token}"):
            pass


def test_sign_predict_sequence_returns_hello():
    # 30 frames of open-palm landmarks (21x3 each) — the real live-call
    # payload shape. Default path with no weights -> 503; demo=true -> demo.
    seq = [_open_palm() for _ in range(30)]
    r = client.post("/api/ai/sign/predict", json={"sequence": seq, "force": True})
    assert r.status_code == 503, r.text
    assert r.json()["detail"]["mode"] == "model-not-trained"
    r = client.post("/api/ai/sign/predict",
                    json={"sequence": seq, "force": True, "demo": True})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["mode"] == "demo-rule-based"
    assert data["prediction"] == "Hello"


def test_transcript_post_without_consent_400():
    headers, _ = _auth_headers()
    room = client.post("/api/rooms", json={"name": "Consent"}, headers=headers).json()
    r = client.post(f"/api/rooms/{room['id']}/transcripts",
                    json={"speaker_name": "A", "direction": "speech-to-text", "text": "Hi"},
                    headers=headers)
    assert r.status_code == 400, r.text


def test_signal_endpoints_require_auth():
    headers, _ = _auth_headers()
    room = client.post("/api/rooms", json={"name": "Sig"}, headers=headers).json()
    r = client.post(f"/api/rooms/{room['id']}/signal",
                    json={"type": "caption", "payload": {"text": "x"}})
    assert r.status_code == 401
    r = client.get(f"/api/rooms/{room['id']}/signal")
    assert r.status_code == 401
    # Authenticated still works (and is capped, not unbounded).
    r = client.post(f"/api/rooms/{room['id']}/signal",
                    json={"type": "caption", "payload": {"text": "x"}}, headers=headers)
    assert r.status_code == 200


# ---------------------------------------------------------------- LiveKit
def _make_room(headers):
    r = client.post("/api/rooms", json={"name": "LK"}, headers=headers)
    assert r.status_code == 201, r.text
    return r.json()


def _decode_livekit_token(token: str) -> dict:
    """Decode a LiveKit JWT with the dev secret (its own HMAC key)."""
    from jose import jwt
    return jwt.decode(token, "secret", algorithms=["HS256"])


def test_livekit_token_is_real_jwt_with_grants():
    headers, _ = _auth_headers()
    room = _make_room(headers)
    r = client.get(f"/api/rooms/{room['id']}/token", headers=headers)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["mode"] == "livekit"
    assert data["url"] == "ws://localhost:7880"
    assert data["code"] == room["code"] and data["room_id"] == room["id"]
    claims = _decode_livekit_token(data["token"])
    assert claims["iss"] == "devkey"
    video = claims["video"]
    assert video["room"] == room["code"]          # room name = room code
    assert video["roomJoin"] is True
    assert video["roomAdmin"] is True            # creator is room admin
    assert video["canPublish"] is True and video["canSubscribe"] is True


def test_livekit_token_non_creator_is_not_admin():
    creator_headers, _ = _auth_headers()
    room = _make_room(creator_headers)
    other_headers, _ = _auth_headers()
    client.post(f"/api/rooms/{room['id']}/join", json={}, headers=other_headers)
    r = client.get(f"/api/rooms/{room['id']}/token", headers=other_headers)
    assert r.status_code == 200, r.text
    claims = _decode_livekit_token(r.json()["token"])
    assert claims["video"]["roomAdmin"] is not True


def test_livekit_token_unconfigured_returns_503(monkeypatch):
    from app import config
    monkeypatch.setattr(config, "LIVEKIT_URL", "")
    monkeypatch.setattr(config, "LIVEKIT_API_KEY", "")
    monkeypatch.setattr(config, "LIVEKIT_API_SECRET", "")
    headers, _ = _auth_headers()
    room = _make_room(headers)
    r = client.get(f"/api/rooms/{room['id']}/token", headers=headers)
    assert r.status_code == 503, r.text
    assert "LiveKit" in r.json()["detail"]


def test_join_records_participant_and_list():
    creator_headers, _ = _auth_headers()
    room = _make_room(creator_headers)
    joiner_headers, joiner_email = _auth_headers()
    r = client.post(f"/api/rooms/{room['id']}/join", json={}, headers=joiner_headers)
    assert r.status_code == 200, r.text
    r = client.get(f"/api/rooms/{room['id']}/participants", headers=creator_headers)
    assert r.status_code == 200, r.text
    rows = r.json()
    assert any(p["name"] == "Test User" and p["is_active"] for p in rows)
    assert len(rows) == 1  # creator has not joined; only the joiner is listed
    # Re-join reactivates instead of duplicating the row.
    client.post(f"/api/rooms/{room['id']}/leave", headers=joiner_headers)
    client.post(f"/api/rooms/{room['id']}/join", json={}, headers=joiner_headers)
    rows = client.get(f"/api/rooms/{room['id']}/participants",
                      headers=creator_headers).json()
    assert len(rows) == 1 and rows[0]["is_active"] is True


def test_leave_marks_participant_inactive():
    creator_headers, _ = _auth_headers()
    room = _make_room(creator_headers)
    joiner_headers, _ = _auth_headers()
    client.post(f"/api/rooms/{room['id']}/join", json={}, headers=joiner_headers)
    r = client.post(f"/api/rooms/{room['id']}/leave", headers=joiner_headers)
    assert r.status_code == 200, r.text
    assert r.json()["is_active"] is False and r.json()["left_at"]
    rows = client.get(f"/api/rooms/{room['id']}/participants",
                      headers=creator_headers).json()
    assert rows[0]["is_active"] is False


def test_leave_with_duplicate_rows_marks_all_and_lists_once():
    """Two simultaneous /join calls can create two rows for one user.
    Leaving must mark both inactive, and the roster lists the person once
    (previously the second row kept them 'in the room' after leaving)."""
    from app.database import SessionLocal
    from app.models import RoomParticipant
    creator_headers, _ = _auth_headers()
    room = _make_room(creator_headers)
    joiner_headers, _ = _auth_headers()
    client.post(f"/api/rooms/{room['id']}/join", json={}, headers=joiner_headers)
    me = client.get("/api/auth/me", headers=joiner_headers).json()
    db = SessionLocal()
    try:
        db.add(RoomParticipant(room_id=room["id"], user_id=me["id"],
                               identity=me["id"], name="dup"))
        db.commit()
    finally:
        db.close()
    rows = client.get(f"/api/rooms/{room['id']}/participants", headers=creator_headers).json()
    assert [r["identity"] for r in rows].count(me["id"]) == 1
    client.post(f"/api/rooms/{room['id']}/leave", headers=joiner_headers)
    rows = client.get(f"/api/rooms/{room['id']}/participants", headers=creator_headers).json()
    mine = [r for r in rows if r["identity"] == me["id"]]
    assert len(mine) == 1 and mine[0]["is_active"] is False


def test_participants_non_member_403():
    creator_headers, _ = _auth_headers()
    room = _make_room(creator_headers)
    stranger_headers, _ = _auth_headers()
    r = client.get(f"/api/rooms/{room['id']}/participants", headers=stranger_headers)
    assert r.status_code == 403, r.text


def test_end_room_permissions_and_state():
    creator_headers, _ = _auth_headers()
    room = _make_room(creator_headers)
    joiner_headers, _ = _auth_headers()
    client.post(f"/api/rooms/{room['id']}/join", json={}, headers=joiner_headers)
    r = client.post(f"/api/rooms/{room['id']}/end", headers=joiner_headers)
    assert r.status_code == 403, r.text
    # No LiveKit server is running in tests: end must still succeed
    # (RoomService delete is best-effort) and mark the room ended.
    r = client.post(f"/api/rooms/{room['id']}/end", headers=creator_headers)
    assert r.status_code == 200, r.text
    assert r.json()["is_active"] is False
    assert r.json()["livekit_room_deleted"] is False
    r = client.get(f"/api/rooms/{room['id']}", headers=creator_headers)
    assert r.json()["is_active"] is False
    rows = client.get(f"/api/rooms/{room['id']}/participants",
                      headers=creator_headers).json()
    assert rows and all(not p["is_active"] for p in rows)
    # Ended rooms refuse new joins.
    r = client.post(f"/api/rooms/{room['id']}/join", json={}, headers=joiner_headers)
    assert r.status_code == 410


def _webhook_auth_token(body: str) -> str:
    """Sign a webhook body the way LiveKit does (HMAC-SHA256 claim)."""
    import base64
    import hashlib
    from livekit import api as lk_api
    digest = base64.b64encode(hashlib.sha256(body.encode()).digest()).decode()
    return lk_api.AccessToken("devkey", "secret").with_sha256(digest).to_jwt()


def test_webhook_participant_joined_and_left():
    import json as _json
    creator_headers, _ = _auth_headers()
    room = _make_room(creator_headers)
    body = _json.dumps({"event": "participant_joined",
                        "room": {"name": room["code"]},
                        "participant": {"identity": "webhook-user",
                                        "name": "Webhook User", "sid": "PA_x"}})
    r = client.post("/api/livekit/webhook", content=body,
                    headers={"Authorization": _webhook_auth_token(body),
                             "Content-Type": "application/json"})
    assert r.status_code == 200, r.text
    assert r.json() == {"received": True, "verified": True,
                        "event": "participant_joined"}
    rows = client.get(f"/api/rooms/{room['id']}/participants",
                      headers=creator_headers).json()
    assert any(p["identity"] == "webhook-user" and p["is_active"] for p in rows)
    body = _json.dumps({"event": "participant_left",
                        "room": {"name": room["code"]},
                        "participant": {"identity": "webhook-user",
                                        "name": "Webhook User", "sid": "PA_x"}})
    r = client.post("/api/livekit/webhook", content=body,
                    headers={"Authorization": _webhook_auth_token(body),
                             "Content-Type": "application/json"})
    assert r.status_code == 200
    rows = client.get(f"/api/rooms/{room['id']}/participants",
                      headers=creator_headers).json()
    assert any(p["identity"] == "webhook-user" and not p["is_active"] for p in rows)


def test_webhook_bad_signature_rejected_malformed_ok():
    import json as _json
    body = _json.dumps({"event": "participant_joined",
                        "room": {"name": "NOPE"},
                        "participant": {"identity": "x", "name": "X"}})
    r = client.post("/api/livekit/webhook", content=body,
                    headers={"Authorization": "Bearer bogus",
                             "Content-Type": "application/json"})
    assert r.status_code == 200
    assert r.json()["received"] is False  # signature failed, no 500
    r = client.post("/api/livekit/webhook", content="{not json",
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 200 and r.json()["received"] is False


# ---------------------------------------------------------------- AI pipelines
def _pose(kind: str):
    """One 21-landmark frame for a demo pose (SYNTHETIC test fixture only)."""
    lm = [[0.5, 0.9, 0.0]] * 21
    lm = [list(p) for p in lm]
    lm[9] = [0.5, 0.7, 0.0]  # middle MCP (scale anchor)
    up = {"Hello": [8, 12, 16, 20], "Yes": [8], "No": [],
          "Help": [8, 12], "Thank You": [], "Emergency": [20]}[kind]
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        lm[pip] = [0.5, 0.6, 0.0]
        lm[tip] = [0.5, 0.4, 0.0] if tip in up else [0.5, 0.8, 0.0]
    # Thumb: up only for Thank You.
    lm[3] = [0.5, 0.6, 0.0]
    lm[4] = [0.5, 0.4, 0.0] if kind == "Thank You" else [0.5, 0.7, 0.0]
    return lm


def _pose_seq(kind: str, n: int = 10, seed: int = 0):
    """n jittered frames of a pose — a SYNTHETIC training fixture."""
    import random
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        frame = _pose(kind)
        out.append([[x + rng.uniform(-0.01, 0.01),
                     y + rng.uniform(-0.01, 0.01), z]
                    for x, y, z in frame])
    return out


def test_sign_status(monkeypatch, tmp_path):
    # Hermetic: point the samples dir at an empty tmp dir, so the test
    # does not depend on ambient state (a sample recorded against the
    # live server now lives in backend/data/sign_samples/).
    from app import config
    monkeypatch.setattr(config, "SIGN_SAMPLES_DIR", str(tmp_path / "samples"))
    r = client.get("/api/ai/sign/status")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["mode"] == "model-not-trained"
    assert data["trained"] is False
    assert data["torch_available"] is True
    assert data["vocabulary"] == ["Hello", "Yes", "No", "Help", "Thank You", "Emergency", "Please", "Sorry", "Good", "Goodbye"]
    assert all(v == 0 for v in data["samples_per_sign"].values())


def test_ai_status_summary():
    r = client.get("/api/ai/status")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["sign"]["mode"] == "model-not-trained"
    assert data["sign"]["trained"] is False
    assert data["sign"]["torch_available"] is True
    assert data["whisper"]["available"] is True
    assert data["whisper"]["model"] == "tiny"
    assert data["whisper"]["engine"] == "whisper-tiny"
    assert data["tts"]["available"] is True
    assert "en_US-lessac-medium" in data["tts"]["voices"]
    assert data["voice_agent"]["enabled"] is True
    # rooms_active is a COUNT (int), not a room-id list — see
    # app/services/voice_agent.py rooms_active().
    assert data["voice_agent"]["rooms_active"] == 0


def test_sign_collect_and_samples(monkeypatch, tmp_path):
    from app import config
    monkeypatch.setattr(config, "SIGN_SAMPLES_DIR", str(tmp_path / "samples"))
    headers, _ = _auth_headers()  # POST /api/ai/sign/collect requires authentication
    seq = _pose_seq("Hello", n=10, seed=1)
    r = client.post("/api/ai/sign/collect", json={"label": "Hello", "sequence": seq},
                    headers=headers)
    assert r.status_code == 201, r.text
    assert r.json()["saved"] is True and r.json()["label"] == "Hello"
    r = client.get("/api/ai/sign/samples")
    assert r.status_code == 200
    assert r.json()["samples_per_sign"]["Hello"] == 1 and r.json()["total"] == 1
    # Invalid label / too-short sequence are rejected.
    r = client.post("/api/ai/sign/collect", json={"label": "Wave", "sequence": seq},
                    headers=headers)
    assert r.status_code == 400
    r = client.post("/api/ai/sign/collect",
                    json={"label": "Hello", "sequence": _pose_seq("Hello", n=4)},
                    headers=headers)
    assert r.status_code == 422


def test_sign_train_job_lifecycle(monkeypatch, tmp_path):
    """End-to-end trained path on SYNTHETIC fixtures (test-only data).

    Seeds 8 jittered sequences per label, trains a small SignLSTM via the
    HTTP job endpoints, then predicts with the trained model.
    """
    import time
    from app import config
    from app.services import sign_data, sign_model_service

    monkeypatch.setattr(config, "SIGN_SAMPLES_DIR", str(tmp_path / "samples"))
    monkeypatch.setattr(config, "SIGN_MODEL_PATH", str(tmp_path / "sign_lstm.pt"))
    # POST /api/ai/sign/train is admin-only: register a user and list
    # their email in ADMIN_EMAILS for the duration of this test.
    admin_headers, admin_email = _auth_headers()
    monkeypatch.setattr(config, "ADMIN_EMAILS", {admin_email})
    for label in ["Hello", "Yes", "No", "Help", "Thank You", "Emergency"]:
        for i in range(8):
            sign_data.save_sample(label, _pose_seq(label, n=10, seed=100 + i))
    assert sign_data.counts()["total"] == 48

    r = client.post("/api/ai/sign/train", json={"epochs": 20, "hidden_dim": 64},
                    headers=admin_headers)
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    # A second train while one runs -> 409.
    r = client.post("/api/ai/sign/train", json={"epochs": 1}, headers=admin_headers)
    assert r.status_code == 409, r.text
    # Poll until done.
    job = None
    for _ in range(180):
        job = client.get(f"/api/ai/sign/train/{job_id}").json()
        if job["status"] in ("done", "failed"):
            break
        time.sleep(1)
    assert job is not None and job["status"] == "done", job
    assert job["epochs_done"] == 20
    assert job["val_accuracy"] is not None and 0.0 <= job["val_accuracy"] <= 1.0
    # The trained model is now live.
    r = client.get("/api/ai/sign/status")
    assert r.json()["trained"] is True and r.json()["mode"] == "trained-model"
    seq = _pose_seq("Hello", n=10, seed=999)
    r = client.post("/api/ai/sign/predict", json={"sequence": seq, "force": True})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["mode"] == "trained-model"
    assert data["prediction"] == "Hello"
    assert data["confidence"] >= 0.5
    sign_model_service.clear_model_cache()


def test_mediapipe_extract_no_hand():
    """A blank image has no hand -> found:false. Skips (clearly) when the
    MediaPipe native lib cannot run in this environment."""
    import cv2
    import numpy as np
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", blank)
    assert ok
    headers, _ = _auth_headers()  # /extract requires auth (P1)
    r = client.post("/api/ai/sign/extract",
                    files={"file": ("blank.png", buf.tobytes(), "image/png")},
                    headers=headers)
    if r.status_code == 503:
        pytest.skip(f"MediaPipe unavailable in this environment: {r.text[:200]}")
    assert r.status_code == 200, r.text
    assert r.json()["found"] is False and r.json()["landmarks"] is None
    from app.services import vision_service
    vision_service.release()


def _piper_wav_bytes(text: str) -> bytes:
    """Synthesise text with Piper and return the WAV bytes (test helper)."""
    from app.services import tts_service
    result = tts_service.synthesize(text)
    audio_id = result["audio_url"].rsplit("/", 1)[-1]
    with open(tts_service.audio_path(audio_id), "rb") as f:
        return f.read()


def test_whisper_transcribe_real_audio():
    """Piper-synthesised speech -> real Whisper transcription."""
    from app.services import speech_service, tts_service
    wav = _piper_wav_bytes("Hello world, this is a test.")
    tts_service.release_voices()  # free RAM before loading Whisper
    headers, _ = _auth_headers()  # POST /api/ai/transcribe/audio requires authentication
    r = client.post("/api/ai/transcribe/audio",
                    files={"file": ("speech.wav", wav, "audio/wav")},
                    headers=headers)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["engine"] == "whisper-tiny"
    assert "hello" in data["text"].lower() and "world" in data["text"].lower()
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["segments"] and data["is_final"] is True
    speech_service.release_model()


def _wav_to_pcm16_16k(wav: bytes) -> bytes:
    """WAV bytes (any rate, mono 16-bit) -> raw 16 kHz PCM16, the format
    the browser streams to /ws/captions."""
    import io as _io
    import wave

    import numpy as np
    with wave.open(_io.BytesIO(wav)) as w:
        rate = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32)
    n_out = int(len(x) * 16000 / rate)
    y = np.interp(np.arange(n_out) * rate / 16000, np.arange(len(x)), x)
    return y.astype("<i2").tobytes()


def test_captions_websocket():
    """WS /ws/captions: pcm16 config + audio + flush -> real Whisper
    caption; client text -> labelled passthrough."""
    from app.services import speech_service, tts_service
    headers, _ = _auth_headers()
    token = headers["Authorization"].split()[1]
    wav = _piper_wav_bytes("Hello world, captions test.")
    tts_service.release_voices()
    pcm = _wav_to_pcm16_16k(wav)
    with client.websocket_connect(f"/ws/captions?token={token}") as ws:
        ws.send_text('{"type": "text", "text": "Hello there"}')
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "caption" and msg["text"] == "Hello there"
        assert msg["engine"] == "client-text-passthrough"
        ws.send_text('{"type": "config", "format": "pcm16", "sample_rate": 16000}')
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "status" and msg["state"] in ("loading", "ready")
        # ~250 ms frames, like the browser's PcmMicStreamer.
        for off in range(0, len(pcm), 8000):
            ws.send_bytes(pcm[off:off + 8000])
        ws.send_text('{"type": "flush"}')
        while True:
            msg = json.loads(ws.receive_text())
            if msg["type"] != "status":
                break
        assert msg["type"] == "caption", msg
        assert msg["engine"] == "whisper-tiny"
        assert "hello" in msg["text"].lower()
        assert msg["latency_s"] >= 0 and msg["audio_s"] > 0.5
    speech_service.release_model()


def _tone_pcm16(seconds: float, amp: float) -> bytes:
    import numpy as np
    t = np.arange(int(seconds * 16000)) / 16000
    return (amp * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2").tobytes()


def test_captions_websocket_stubbed_whisper(monkeypatch):
    """Socket protocol without the model: each utterance (cut at a
    pause) is transcribed once, in order — the call's earlier audio is
    never re-sent to Whisper."""
    from app.services import speech_service
    calls = []

    async def fake_transcribe(samples, language="en"):
        calls.append(len(samples) / 16000)
        return {"text": f"utterance {len(calls)}", "confidence": 0.8,
                "engine": "whisper-tiny"}

    monkeypatch.setattr(speech_service, "transcribe_pcm_async", fake_transcribe)
    monkeypatch.setattr(speech_service, "model_loaded", lambda: True)
    headers, _ = _auth_headers()
    token = headers["Authorization"].split()[1]
    audio = (_tone_pcm16(0.5, 0.0) + _tone_pcm16(1.5, 0.3) + _tone_pcm16(1.0, 0.0)
             + _tone_pcm16(1.5, 0.3) + _tone_pcm16(1.0, 0.0))
    with client.websocket_connect(f"/ws/captions?token={token}") as ws:
        ws.send_text('{"type": "config", "format": "pcm16", "sample_rate": 16000}')
        assert json.loads(ws.receive_text()) == {"type": "status", "state": "ready"}
        for off in range(0, len(audio), 8000):
            ws.send_bytes(audio[off:off + 8000])
        texts = [json.loads(ws.receive_text())["text"] for _ in range(2)]
    assert texts == ["utterance 1", "utterance 2"]
    assert len(calls) == 2 and all(c < 3.0 for c in calls)


def test_captions_websocket_rejects_audio_without_config():
    """Old clients that stream webm without a pcm16 config get a
    machine-readable error (they then switch to browser captions)."""
    headers, _ = _auth_headers()
    token = headers["Authorization"].split()[1]
    with client.websocket_connect(f"/ws/captions?token={token}") as ws:
        ws.send_bytes(bytes([0x1A, 0x45, 0xDF, 0xA3]) + bytes(100))  # webm magic
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "error" and msg["code"] == "bad-format"
        ws2_bad = '{"type": "config", "format": "pcm16", "sample_rate": 48000}'
    with client.websocket_connect(f"/ws/captions?token={token}") as ws:
        ws.send_text(ws2_bad)
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "error" and msg["code"] == "bad-format"


def _ensure_livekit_server():
    """Start the local LiveKit dev server if :7880 is not listening.
    Returns the Popen object, or None if a server was already running
    (in which case the caller must NOT stop it)."""
    import socket
    import subprocess
    s = socket.socket()
    try:
        s.connect(("127.0.0.1", 7880))
        s.close()
        return None
    except OSError:
        pass
    finally:
        s.close()
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    livekit_dir = os.path.join(os.path.dirname(backend_dir), "livekit")
    server_path = os.path.join(livekit_dir, "livekit-server")
    if not os.path.isfile(server_path):
        pytest.skip(
            "LiveKit server binary is not installed; install it to run the "
            "voice-agent lifecycle test.")
    proc = subprocess.Popen(
        [server_path, "--config", "livekit.yaml"],
        cwd=livekit_dir,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(40):
        s = socket.socket()
        try:
            s.connect(("127.0.0.1", 7880))
            s.close()
            return proc
        except OSError:
            pass
        finally:
            s.close()
        import time
        time.sleep(0.5)
    proc.terminate()
    raise RuntimeError("LiveKit server did not start on :7880")


def test_voice_agent_audio_pipeline():
    """The agent's real audio path, without a LiveKit room: Piper synthesis
    -> 22050 Hz WAV -> resample to 48 kHz -> 10 ms frames streamed to the
    audio source. Verifies the resample and framing for real."""
    import asyncio
    from app.services import tts_service, voice_agent

    class FakeSource:
        def __init__(self):
            self.frames = []

        async def capture_frame(self, frame):
            self.frames.append(frame)

    agent = voice_agent.VoiceAgent("room-test", "ABCDEF")
    source = FakeSource()
    result = {"spoken": False, "duration_s": 0.0, "error": None}
    try:
        asyncio.run(agent._speak_one(source, "Hello world", result))
    finally:
        tts_service.release_voices()
    assert result["spoken"] is True, result
    assert result["duration_s"] > 0
    # ~100 frames per second of audio, each 10 ms (480 int16 samples).
    assert abs(len(source.frames) - result["duration_s"] * 100) <= 2
    for frame in source.frames:
        assert frame.sample_rate == 48000 and frame.num_channels == 1
        assert frame.samples_per_channel == 480
        # data is an int16 memoryview: 480 elements == 960 bytes.
        assert frame.data.nbytes == 960


def test_voice_agent_lifecycle():
    """Voice agent joins the local LiveKit room, speaks, stops cleanly.

    SANDBOX LIMITATION (2026-11-17): the livekit-rtc (Rust) client's
    room.connect() never completes in this sandbox — the server log shows
    the join sessions starting, but the client keeps retrying the signal
    connection and times out (same family as the documented UDP/ICE
    block: the Rust engine's connect path cannot finish here, while the
    browser JS client connected fine). The agent reports this honestly
    as 503 instead of faking a join. When that happens this test SKIPS
    with this explanation; run it on a normal machine/network to verify
    the full join/speak/stop lifecycle.
    """
    from app.services import tts_service
    proc = _ensure_livekit_server()
    try:
        headers, _ = _auth_headers()
        room = _make_room(headers)
        client.post(f"/api/rooms/{room['id']}/join", json={}, headers=headers)
        r = client.post(f"/api/ai/rooms/{room['id']}/voice/start", headers=headers,
                        timeout=60)
        if r.status_code == 503:
            pytest.skip(f"Voice agent cannot join LiveKit in this sandbox: "
                        f"{r.json().get('detail', '')[:200]} "
                        "(livekit-rtc connect never completes here; verify on a "
                        "normal machine.)")
        assert r.status_code == 200, r.text
        assert r.json()["running"] is True
        assert r.json()["identity"] == "signspeak-voice"
        r = client.post(f"/api/ai/rooms/{room['id']}/voice/start", headers=headers,
                        timeout=60)
        assert r.status_code == 409, r.text
        r = client.post(f"/api/ai/rooms/{room['id']}/voice/speak",
                        json={"text": "Hello"}, headers=headers, timeout=120)
        assert r.status_code == 200, r.text
        assert r.json()["spoken"] is True and r.json()["duration_s"] > 0
        r = client.get(f"/api/ai/rooms/{room['id']}/voice/status", headers=headers)
        assert r.status_code == 200 and r.json()["running"] is True
        r = client.post(f"/api/ai/rooms/{room['id']}/voice/stop", headers=headers,
                        timeout=60)
        assert r.status_code == 200, r.text
        assert r.json()["running"] is False
        # Ending the room also stops any agent (best-effort).
        r = client.post(f"/api/rooms/{room['id']}/end", headers=headers)
        assert r.status_code == 200, r.text
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
        tts_service.release_voices()


def test_voice_status_without_agent():
    """GET /voice/status on a room with no agent must return 200
    (running=false) — no LiveKit join needed. Regression test: the
    endpoint once referenced an undefined `room` variable and 500'd
    on every call (found by manual verification 2026-11-17)."""
    headers, _ = _auth_headers()
    room = _make_room(headers)
    client.post(f"/api/rooms/{room['id']}/join", json={}, headers=headers)
    r = client.get(f"/api/ai/rooms/{room['id']}/voice/status", headers=headers)
    assert r.status_code == 200, r.text
    assert r.json()["running"] is False
    assert r.json()["identity"] == "signspeak-voice"
    assert r.json()["room_id"] == room["id"]
    # And a non-member gets 403, not a 500.
    other_headers, _ = _auth_headers()
    r = client.get(f"/api/ai/rooms/{room['id']}/voice/status", headers=other_headers)
    assert r.status_code == 403, r.text
