"""In-process room state + WebSocket fan-out.

Deliberately simple, and deliberately documented as a single-instance design.
The moment you run two API replicas behind a load balancer, two clients in the
same call can land on different processes and stop seeing each other's
captions. The fix is the Redis pub/sub path below, which is wired but off by
default because it is one more thing to run on a laptop.

Do not discover this in the demo. Either run one instance, or set
REDIS_FANOUT=1 and run the pub/sub bridge.
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict

from fastapi import WebSocket

from .services.ordering import ClockSync

_clocks: dict[str, ClockSync] = defaultdict(ClockSync)


def clock_for(identity: str) -> ClockSync:
    return _clocks[identity]


class SessionRegistry:
    def __init__(self) -> None:
        self._rooms: dict[str, set[WebSocket]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def join(self, room: str, ws: WebSocket) -> None:
        async with self._lock:
            self._rooms[room].add(ws)

    async def leave(self, room: str, ws: WebSocket) -> None:
        async with self._lock:
            self._rooms[room].discard(ws)
            if not self._rooms[room]:
                self._rooms.pop(room, None)

    async def broadcast(self, room: str, message: dict, exclude: WebSocket | None = None) -> None:
        async with self._lock:
            targets = [w for w in self._rooms.get(room, set()) if w is not exclude]
        payload = json.dumps(message)
        dead = []
        for ws in targets:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.leave(room, ws)

    def count(self, room: str) -> int:
        return len(self._rooms.get(room, set()))


sessions = SessionRegistry()
