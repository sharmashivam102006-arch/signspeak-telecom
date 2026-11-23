"""Room WebSocket: captions, presence, and recognition status.

This channel carries TEXT ONLY. Audio and video go through LiveKit; landmarks
never leave the browser. If you ever find yourself about to send a frame or a
landmark array down this socket, stop - that is the privacy design failing, and
it will fail quietly.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from ..state import clock_for, sessions

router = APIRouter()


@router.websocket("/ws/rooms/{room_name}")
async def room_socket(
    websocket: WebSocket,
    room_name: str,
    identity: str = Query(...),
) -> None:
    await websocket.accept()
    room = room_name.upper().strip()
    await sessions.join(room, websocket)
    await sessions.broadcast(
        room, {"type": "presence", "identity": identity, "event": "joined",
               "count": sessions.count(room)}
    )

    try:
        while True:
            msg = await websocket.receive_json()
            kind = msg.get("type")

            if kind == "ping":
                # Clock sync over the socket, so the offset stays fresh as
                # network conditions drift mid-call.
                t2 = time.monotonic() * 1000.0
                await websocket.send_json(
                    {"type": "pong", "t1": msg.get("t1"), "t2": t2,
                     "t3": time.monotonic() * 1000.0}
                )

            elif kind == "clock_result":
                clock_for(identity).observe(
                    msg["t1"], msg["t2"], msg["t3"], msg["t4"]
                )

            elif kind in {"caption", "sign_status", "typing", "correction"}:
                msg["identity"] = identity
                msg.setdefault("server_ms", time.monotonic() * 1000.0)
                await sessions.broadcast(room, msg, exclude=websocket)

    except WebSocketDisconnect:
        pass
    finally:
        await sessions.leave(room, websocket)
        await sessions.broadcast(
            room, {"type": "presence", "identity": identity, "event": "left",
                   "count": sessions.count(room)}
        )
