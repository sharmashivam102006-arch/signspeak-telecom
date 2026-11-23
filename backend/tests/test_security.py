"""Security tests — negative tests for every hole fixed in the
2026-11-18 Priority-1 security pass (see audit report
~/workspace/signspeak-telecom-audit/AUDIT-REPORT-2026-11-17.md).

Run: python -m pytest tests/test_api.py tests/test_security.py -v
(from backend/). Same env setup as tests/test_api.py: the test DB,
test secret and LiveKit dev credentials are set before importing the
app, so these tests work standalone too.
"""
import json
import os
import sys
import time

os.environ["DATABASE_URL"] = "sqlite:///./test_signspeak.db"
os.environ["SECRET_KEY"] = "test-secret"
os.environ["LIVEKIT_URL"] = "ws://localhost:7880"
os.environ["LIVEKIT_API_KEY"] = "devkey"
os.environ["LIVEKIT_API_SECRET"] = "secret"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient

from app.database import Base, engine

Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)

from app.main import app  # noqa: E402
from app.security import reset_rate_limits  # noqa: E402

client = TestClient(app)


def _auth_headers():
    import uuid as _uuid
    email = f"sec-{_uuid.uuid4().hex[:8]}@example.com"
    r = client.post("/api/auth/register", json={
        "email": email, "name": "Sec User", "password": "password123"})
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}, email


def _guest_headers(name="Sec Guest"):
    r = client.post("/api/auth/guest", json={"name": name})
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def _make_room(headers, name="Sec Room"):
    r = client.post("/api/rooms", json={"name": name}, headers=headers)
    assert r.status_code == 201, r.text
    return r.json()


def _decode_livekit_token(token: str) -> dict:
    from jose import jwt
    return jwt.decode(token, "secret", algorithms=["HS256"])


# ---------------------------------------------------------------- rooms
def test_room_lookup_requires_auth():
    headers, _ = _auth_headers()
    room = _make_room(headers)
    # No token at all -> 401 (was 200 with the full room + invite link).
    r = client.get(f"/api/rooms/{room['code']}")
    assert r.status_code == 401, r.text
    # Unknown room without auth -> 401 as well (auth runs first).
    r = client.get("/api/rooms/XXXXXX")
    assert r.status_code == 401, r.text


def test_room_lookup_non_member_403_member_200():
    creator_headers, _ = _auth_headers()
    room = _make_room(creator_headers)
    stranger_headers, _ = _auth_headers()
    r = client.get(f"/api/rooms/{room['id']}", headers=stranger_headers)
    assert r.status_code == 403, r.text
    # Creator can see it; a participant who joined can too.
    r = client.get(f"/api/rooms/{room['id']}", headers=creator_headers)
    assert r.status_code == 200, r.text
    joiner_headers, _ = _auth_headers()
    client.post(f"/api/rooms/{room['id']}/join", json={}, headers=joiner_headers)
    r = client.get(f"/api/rooms/{room['id']}", headers=joiner_headers)
    assert r.status_code == 200, r.text


def test_token_non_member_403():
    creator_headers, _ = _auth_headers()
    room = _make_room(creator_headers)
    # Fresh guest who never joined -> 403 (was 200 with a valid token).
    fresh_guest = _guest_headers("Never Joined")
    r = client.get(f"/api/rooms/{room['id']}/token", headers=fresh_guest)
    assert r.status_code == 403, r.text
    # Creator and a joined participant still get tokens.
    r = client.get(f"/api/rooms/{room['id']}/token", headers=creator_headers)
    assert r.status_code == 200, r.text
    joiner_headers, _ = _auth_headers()
    client.post(f"/api/rooms/{room['id']}/join", json={}, headers=joiner_headers)
    r = client.get(f"/api/rooms/{room['id']}/token", headers=joiner_headers)
    assert r.status_code == 200, r.text


def test_token_unknown_room_404_and_ended_room_410():
    headers, _ = _auth_headers()
    r = client.get("/api/rooms/XXXXXX/token", headers=headers)
    assert r.status_code == 404, r.text
    room = _make_room(headers)
    client.post(f"/api/rooms/{room['id']}/end", headers=headers)
    r = client.get(f"/api/rooms/{room['id']}/token", headers=headers)
    assert r.status_code == 410, r.text


def test_token_ttl_is_configured_value():
    """Issued LiveKit token: exp > nbf and exp - nbf == the configured
    LIVEKIT_TOKEN_TTL_HOURS (default 6h). The SDK computes nbf and exp
    from two separate now() calls truncated to seconds, so allow ±1s."""
    from app import config
    headers, _ = _auth_headers()
    room = _make_room(headers)
    r = client.get(f"/api/rooms/{room['id']}/token", headers=headers)
    assert r.status_code == 200, r.text
    claims = _decode_livekit_token(r.json()["token"])
    assert claims["exp"] > claims["nbf"]
    expected = config.LIVEKIT_TOKEN_TTL_HOURS * 3600
    assert abs((claims["exp"] - claims["nbf"]) - expected) <= 1, claims


def test_jwt_access_token_expiry_is_24h():
    """The app JWT keeps its 24h expiry (config.ACCESS_TOKEN_EXPIRE_HOURS
    = 24, unchanged by this pass)."""
    from app import config
    assert config.ACCESS_TOKEN_EXPIRE_HOURS == 24
    headers, _ = _auth_headers()
    token = headers["Authorization"].split()[1]
    from jose import jwt
    claims = jwt.decode(token, "test-secret", algorithms=["HS256"])
    # exp is ~24h from issue; allow a few minutes of test-runtime skew.
    assert abs((claims["exp"] - time.time()) - 24 * 3600) < 300, claims


def test_signal_endpoints_require_membership():
    creator_headers, _ = _auth_headers()
    room = _make_room(creator_headers)
    stranger_headers, _ = _auth_headers()
    # Non-member POST and GET -> 403 (was 200).
    r = client.post(f"/api/rooms/{room['id']}/signal",
                    json={"type": "caption", "payload": {"text": "x"}},
                    headers=stranger_headers)
    assert r.status_code == 403, r.text
    r = client.get(f"/api/rooms/{room['id']}/signal", headers=stranger_headers)
    assert r.status_code == 403, r.text
    # Creator (member) can still post and read.
    r = client.post(f"/api/rooms/{room['id']}/signal",
                    json={"type": "caption", "payload": {"text": "hi"}},
                    headers=creator_headers)
    assert r.status_code == 200, r.text
    r = client.get(f"/api/rooms/{room['id']}/signal", headers=creator_headers)
    assert r.status_code == 200 and len(r.json()["messages"]) == 1, r.text


# ---------------------------------------------------------------- AI endpoints
def test_tts_requires_auth():
    # Unauthenticated -> 401 (was 200 with a real WAV URL).
    r = client.post("/api/ai/tts", json={"text": "Hello"})
    assert r.status_code == 401, r.text
    headers, _ = _auth_headers()
    r = client.post("/api/ai/tts", json={"text": "Hi"}, headers=headers)
    assert r.status_code == 200, r.text
    assert r.json()["audio_url"].startswith("/api/ai/tts/audio/")


def test_transcribe_audio_requires_auth():
    # Unauthenticated -> 401 (was an open Whisper compute endpoint).
    r = client.post("/api/ai/transcribe/audio",
                    files={"file": ("speech.wav", b"RIFF....", "audio/wav")})
    assert r.status_code == 401, r.text
    # Authenticated with an empty file reaches the endpoint (400), i.e.
    # auth passed and the endpoint validated the payload.
    headers, _ = _auth_headers()
    r = client.post("/api/ai/transcribe/audio",
                    files={"file": ("empty.wav", b"", "audio/wav")},
                    headers=headers)
    assert r.status_code == 400, r.text


def test_extract_requires_auth():
    """POST /api/ai/sign/extract requires authentication. NOTE: the
    endpoint lives in app/routes/sign_recognition.py, owned by the
    parallel P2 agent — this test pins the intended end-state; if it
    fails, the auth change has not landed there yet."""
    import cv2
    import numpy as np
    blank = np.zeros((64, 64, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", blank)
    assert ok
    r = client.post("/api/ai/sign/extract",
                    files={"file": ("blank.png", buf.tobytes(), "image/png")})
    assert r.status_code == 401, r.text


# ---------------------------------------------------------------- webhook
def test_webhook_unconfigured_fails_closed(monkeypatch):
    """With LiveKit unconfigured the webhook must REJECT (503), not
    accept unverified events (previously: accepted with a warning, so
    a forged room_finished could kill any room)."""
    from app import config
    monkeypatch.setattr(config, "LIVEKIT_URL", "")
    monkeypatch.setattr(config, "LIVEKIT_API_KEY", "")
    monkeypatch.setattr(config, "LIVEKIT_API_SECRET", "")
    body = json.dumps({"event": "room_finished",
                       "room": {"name": "ANYCODE"}, "participant": {}})
    r = client.post("/api/livekit/webhook", content=body,
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 503, r.text
    assert "cannot" in r.json()["detail"].lower() or "verif" in r.json()["detail"].lower()


def test_webhook_bad_signature_still_rejected_when_configured():
    """Configured + bad signature -> 200 {received: false} (no retry
    storm, no DB mutation)."""
    body = json.dumps({"event": "room_finished",
                       "room": {"name": "ANYCODE"}, "participant": {}})
    r = client.post("/api/livekit/webhook", content=body,
                    headers={"Authorization": "Bearer bogus",
                             "Content-Type": "application/json"})
    assert r.status_code == 200, r.text
    assert r.json()["received"] is False


# ---------------------------------------------------------------- admin dependency
def test_get_admin_user_contract():
    """get_admin_user: non-admin -> 403; email in ADMIN_EMAILS -> ok;
    id in ADMIN_USER_IDS -> ok. (No admin route exists yet in this
    pass; POST /api/ai/sign/train will use this dependency.)"""
    import uuid as _uuid
    from fastapi import HTTPException

    from app.deps import get_admin_user
    from app.models import User

    user = User(id=str(_uuid.uuid4()), email="plain@example.com",
                name="Plain", hashed_password="")
    with pytest.raises(HTTPException) as excinfo:
        get_admin_user(user)
    assert excinfo.value.status_code == 403

    from app import config
    admin = User(id=str(_uuid.uuid4()), email="ops@example.com",
                 name="Ops", hashed_password="")
    config.ADMIN_EMAILS = {"ops@example.com"}
    config.ADMIN_USER_IDS = set()
    try:
        assert get_admin_user(admin) is admin
        # Same user, but authorised by id instead of email.
        config.ADMIN_EMAILS = set()
        config.ADMIN_USER_IDS = {admin.id}
        assert get_admin_user(admin) is admin
    finally:
        config.ADMIN_EMAILS = set()
        config.ADMIN_USER_IDS = set()


# ---------------------------------------------------------------- CORS
def test_cors_allowlist_has_no_wildcard():
    from app import config
    origins = config.get_cors_origins()
    assert "*" not in origins
    assert all(o.startswith(("http://", "https://")) for o in origins)
    assert "http://localhost:3000" in origins
    # Live preflight check: an unlisted origin gets no allow-origin echo.
    r = client.options("/api/rooms",
                       headers={"Origin": "https://evil.example.com",
                                "Access-Control-Request-Method": "POST"})
    assert r.headers.get("access-control-allow-origin") != "https://evil.example.com"
    # A listed origin does.
    r = client.options("/api/rooms",
                       headers={"Origin": "http://localhost:3000",
                                "Access-Control-Request-Method": "POST"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"


# ---------------------------------------------------------------- exception handlers
def test_exception_handlers_return_json():
    """The global handler returns JSON 500 (not Starlette plain text);
    the rate-limit handler returns JSON 429."""
    import asyncio

    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address
    from slowapi.wrappers import LimitGroup

    from app.main import rate_limit_exceeded_handler, unhandled_exception_handler

    class _Req:
        method = "GET"
        url = type("U", (), {"path": "/x"})()

    resp = asyncio.run(unhandled_exception_handler(_Req(), RuntimeError("boom")))
    assert resp.status_code == 500
    assert json.loads(resp.body) == {"detail": "Internal server error"}
    # Build the exception exactly the way slowapi does (a Limit parsed
    # from a limit string, not a bare string): RateLimitExceeded takes
    # a slowapi Limit object and sets detail from it.
    limit = list(LimitGroup("10/minute", get_remote_address, None, False,
                            None, None, None, 1, False))[0]
    exc = RateLimitExceeded(limit)
    resp = asyncio.run(rate_limit_exceeded_handler(_Req(), exc))
    assert resp.status_code == 429
    assert "Rate limit exceeded" in json.loads(resp.body)["detail"]


# ---------------------------------------------------------------- rate limiting
def test_rate_limit_exceeded_429():
    """POST /api/auth/login is limited to 10/min per IP: the 11th
    attempt in a minute answers 429. Runs last and resets the shared
    in-memory counters before and after, because all TestClient
    requests share one client IP with the rest of the suite."""
    reset_rate_limits()
    try:
        statuses = []
        for _ in range(11):
            r = client.post("/api/auth/login", json={
                "email": "nobody@example.com", "password": "wrongpass"})
            statuses.append(r.status_code)
        assert statuses[:10] == [401] * 10, statuses
        assert statuses[10] == 429, statuses
        assert "Rate limit exceeded" in r.json()["detail"]
    finally:
        reset_rate_limits()
