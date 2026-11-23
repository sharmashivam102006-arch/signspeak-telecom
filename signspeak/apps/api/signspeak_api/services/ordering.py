"""Transcript ordering.

A problem the original design does not mention, and which makes a live demo
look broken within about ninety seconds.

Sign and speech events reach the server on wildly different delays:

    speech  streaming ASR emits partials almost immediately; the final lands
            ~200-400ms after the speaker stops
    sign    the segmenter cannot know a sign ended until it has seen ~280ms of
            stillness, then classification and the round trip add more

So a sign that *started* at t=10.0s can arrive after a spoken word that started
at t=10.6s. Append in arrival order and the transcript reads:

    10:31  Hearing user   How are you?
    10:31  Sign user      Hello

which is the conversation backwards. Worse, the demo script in the spec has
exactly this alternating structure, so the bug shows up in the one recording
that gets shown to an examiner.

The fix is standard: timestamp at *utterance start* on the client, translate
client clocks to a common timeline, and hold each entry in a small buffer long
enough for a late-but-earlier event to overtake it.

Clock translation matters. `performance.now()` on two laptops has no shared
origin, so the client sends its own monotonic timestamp and the server keeps a
per-participant offset estimated the same way NTP does, from the round trip.
Without this, ordering by "client time" orders by whose laptop booted first.
"""

from __future__ import annotations

import asyncio
import heapq
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field


def now_ms() -> float:
    return time.monotonic() * 1000.0


@dataclass
class ClockSync:
    """Per-participant offset between client monotonic time and server time.

    Estimated from ping/pong: the client stamps t1, the server stamps t2 on
    receipt and t3 on send, the client stamps t4. Offset is the usual NTP
    estimator. We keep the sample with the lowest round trip seen, because the
    minimum-delay sample is the least contaminated by queueing.
    """

    offset_ms: float = 0.0
    best_rtt_ms: float = float("inf")
    samples: int = 0

    def observe(self, t1: float, t2: float, t3: float, t4: float) -> None:
        rtt = (t4 - t1) - (t3 - t2)
        if rtt < 0:
            return
        if rtt < self.best_rtt_ms:
            self.best_rtt_ms = rtt
            self.offset_ms = ((t2 - t1) + (t3 - t4)) / 2.0
        self.samples += 1

    def to_server(self, client_ms: float) -> float:
        return client_ms + self.offset_ms

    @property
    def confident(self) -> bool:
        return self.samples >= 3 and self.best_rtt_ms < 400


@dataclass(order=True)
class _Queued:
    sort_key: tuple[float, int]
    entry: dict = field(compare=False)
    release_at: float = field(compare=False, default=0.0)


class ReorderBuffer:
    """Holds entries for `window_ms`, releasing them in start-time order.

    The window is a direct latency-for-correctness trade. 900ms comfortably
    covers the segmentation lag, at the cost of the transcript trailing the
    conversation by about a second. That is the right trade for a *transcript*
    and the wrong one for *live captions*, so captions bypass this entirely and
    render the instant they arrive - being briefly out of order on screen is
    much less confusing than being a second late. Two different consumers, two
    different policies, one source of events.
    """

    def __init__(
        self,
        window_ms: int,
        flush: Callable[[dict], Awaitable[None]],
    ) -> None:
        self.window_ms = window_ms
        self._flush = flush
        self._heap: list[_Queued] = []
        self._seq = 0
        self._task: asyncio.Task | None = None
        self._wake = asyncio.Event()
        self._closed = False

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def push(self, entry: dict, started_at_server_ms: float) -> None:
        self._seq += 1
        item = _Queued(
            sort_key=(started_at_server_ms, self._seq),
            entry=entry,
            release_at=now_ms() + self.window_ms,
        )
        heapq.heappush(self._heap, item)
        self._wake.set()

    async def drain(self) -> None:
        """Flush everything immediately. Called when the call ends - no point
        holding the last sentence back for a window nobody is waiting on."""
        while self._heap:
            await self._flush(heapq.heappop(self._heap).entry)

    async def close(self) -> None:
        self._closed = True
        self._wake.set()
        if self._task:
            await self._task
        await self.drain()

    async def _run(self) -> None:
        while not self._closed:
            if not self._heap:
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass
                continue

            head = self._heap[0]
            delay = head.release_at - now_ms()
            if delay > 0:
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=delay / 1000.0)
                    self._wake.clear()
                except asyncio.TimeoutError:
                    pass
                continue

            await self._flush(heapq.heappop(self._heap).entry)


def merge_partials(entries: list[dict]) -> list[dict]:
    """Collapse streaming ASR partials into finals.

    Streaming recognition emits a growing prefix: "How", "How are", "How are
    you?". Persisting all of them produces a transcript three times too long.
    Keep the last partial per (participant, utterance_id) and drop it entirely
    once the final for that utterance arrives.
    """
    finals = {
        (e["participant_identity"], e.get("utterance_id"))
        for e in entries
        if e.get("is_final")
    }
    latest_partial: dict[tuple, dict] = {}
    out: list[dict] = []
    for e in entries:
        key = (e["participant_identity"], e.get("utterance_id"))
        if e.get("is_final"):
            out.append(e)
        elif key not in finals:
            latest_partial[key] = e
    out.extend(latest_partial.values())
    return sorted(out, key=lambda e: (e["started_at_ms"], e.get("ended_at_ms", 0)))
