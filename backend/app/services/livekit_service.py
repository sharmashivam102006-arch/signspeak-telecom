"""LiveKit integration — REAL tokens and RoomService calls.

Uses the `livekit-api` SDK (a required dependency, see requirements.txt):

* ``generate_token`` issues a real LiveKit AccessToken JWT (room name =
  room code, identity = user id, grants room_join=True, room_admin=True
  for the room creator).
* ``list_participants`` / ``delete_room`` call the LiveKit RoomService
  (the SDK client is async; the sync wrappers below run it with
  ``asyncio.run`` — safe from sync FastAPI endpoints, which run in a
  threadpool with no event loop).
* ``verify_webhook`` validates LiveKit webhook signatures with the SDK's
  ``WebhookReceiver``.

If LIVEKIT_URL / API_KEY / API_SECRET are unset, there is NO silent
fallback to WebRTC P2P: ``generate_token`` returns mode="unconfigured"
with token=None, and the token route answers HTTP 503. The legacy
browser-P2P WebSocket in app/main.py (/ws/call) remains only as a relay
for captions/sign messages; it is not a media path.
"""
import asyncio
import logging
from datetime import timedelta
from typing import Optional

from app import config

logger = logging.getLogger(__name__)

MODE_LIVEKIT = "livekit"
MODE_UNCONFIGURED = "unconfigured"
# Kept for backwards compatibility with older callers/tests.
MODE_P2P = "webrtc-p2p"


def livekit_configured() -> bool:
    return bool(config.LIVEKIT_URL and config.LIVEKIT_API_KEY and config.LIVEKIT_API_SECRET)


def generate_token(room_name: str, identity: str, name: str = "",
                   room_admin: bool = False) -> dict:
    """Return {token, url, mode, note}. Real LiveKit JWT when configured.

    When unconfigured this returns mode="unconfigured", token=None — the
    caller (token route) must turn that into HTTP 503. It never returns
    a fake or P2P token.
    """
    if not livekit_configured():
        return {
            "token": None,
            "url": None,
            "mode": MODE_UNCONFIGURED,
            "note": (
                "LiveKit is not configured (LIVEKIT_URL / LIVEKIT_API_KEY / "
                "LIVEKIT_API_SECRET unset). Start a local server with "
                "`./livekit-server --dev` (see backend/livekit-dev.md) or set "
                "real LiveKit Cloud credentials."
            ),
        }
    from livekit import api as lk_api  # type: ignore

    token = (
        lk_api.AccessToken(config.LIVEKIT_API_KEY, config.LIVEKIT_API_SECRET)
        .with_identity(identity)
        .with_name(name or identity)
        .with_ttl(timedelta(hours=config.LIVEKIT_TOKEN_TTL_HOURS))
        .with_grants(lk_api.VideoGrants(
            room_join=True, room=room_name, room_admin=room_admin))
        .to_jwt()
    )
    return {"token": token, "url": config.LIVEKIT_URL, "mode": MODE_LIVEKIT, "note": ""}


def _api_client():
    """Build a LiveKitAPI client (async SDK). Caller must aclose() it."""
    from livekit import api as lk_api  # type: ignore
    import aiohttp

    return lk_api.LiveKitAPI(
        url=config.LIVEKIT_URL,
        api_key=config.LIVEKIT_API_KEY,
        api_secret=config.LIVEKIT_API_SECRET,
        timeout=aiohttp.ClientTimeout(total=5),
    )


async def _list_participants_async(room_name: str) -> list:
    from livekit import api as lk_api  # type: ignore

    client = _api_client()
    try:
        resp = await client.room.list_participants(
            lk_api.ListParticipantsRequest(room=room_name))
        return [{
            "identity": p.identity,
            "name": p.name or p.identity,
            "sid": p.sid,
            "state": str(p.state),
        } for p in resp.participants]
    finally:
        await client.aclose()


def list_participants(room_name: str) -> list:
    """Participants currently in the LiveKit room. [] on any error."""
    if not livekit_configured():
        return []
    try:
        return asyncio.run(_list_participants_async(room_name))
    except Exception as exc:  # LiveKit down / unreachable — best effort
        logger.warning("LiveKit list_participants(%s) failed: %s", room_name, exc)
        return []


async def _delete_room_async(room_name: str) -> bool:
    from livekit import api as lk_api  # type: ignore

    client = _api_client()
    try:
        await client.room.delete_room(lk_api.DeleteRoomRequest(room=room_name))
        return True
    finally:
        await client.aclose()


def delete_room(room_name: str) -> bool:
    """Delete the LiveKit room. Returns False (never raises) on failure."""
    if not livekit_configured():
        return False
    try:
        return asyncio.run(_delete_room_async(room_name))
    except Exception as exc:  # LiveKit down — caller must not 500
        logger.warning("LiveKit delete_room(%s) failed: %s", room_name, exc)
        return False


def verify_webhook(body: str, auth_token: str):
    """Verify a LiveKit webhook request with the SDK's WebhookReceiver.

    Returns the parsed WebhookEvent on success; raises on bad signature /
    hash mismatch. Uses LIVEKIT_WEBHOOK_SECRET when set, else the API
    secret (LiveKit signs webhooks with the API key/secret by default).
    Raises RuntimeError when LiveKit is not configured at all.
    """
    if not livekit_configured():
        raise RuntimeError("LiveKit is not configured; cannot verify webhook")
    from livekit import api as lk_api  # type: ignore

    secret = config.LIVEKIT_WEBHOOK_SECRET or config.LIVEKIT_API_SECRET
    receiver = lk_api.WebhookReceiver(
        lk_api.TokenVerifier(config.LIVEKIT_API_KEY, secret))
    return receiver.receive(body, auth_token)


def room_url(room_code: str) -> str:
    # Frontend route is /room/[id] — /call/{code} does not exist.
    return f"{config.FRONTEND_URL.rstrip('/')}/room/{room_code}"
