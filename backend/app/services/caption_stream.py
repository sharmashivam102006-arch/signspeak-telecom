"""Live captions over WS /ws/captions — utterance segmentation + a
transcription worker that never falls silently behind.

Why this exists (2026-11-23): the old socket kept every audio byte of the
call and re-transcribed the WHOLE buffer every 2 s. Cost grew with call
length, transcriptions ran inside the receive loop so new audio queued
behind them, and on a small CPU the first caption took ~30 s and the
rest never came. Now:

* The client sends raw 16 kHz mono PCM16 (after a
  ``{"type":"config","format":"pcm16","sample_rate":16000}`` message).
  No container format, so the server can cut the stream anywhere.
* ``SpeechSegmenter`` cuts the stream into utterances with a cheap
  energy detector (adaptive noise floor): an utterance ends after a
  pause, or at ``max_segment_s`` (cut at the quietest recent frame so
  words are rarely split). Silence is dropped — Whisper never sees it.
* ``CaptionSession`` transcribes utterances one at a time in a worker
  task, off the receive loop. Each Whisper call costs the same no matter
  how long the call has been running.
* If captions fall more than ``max_lag_s`` behind the speaker (or the
  model does not load in time), the session sends
  ``{"type":"error","code":"too-slow"}`` once and stops, so the client
  can switch to browser captions instead of showing nothing.
"""
from __future__ import annotations

import asyncio
import collections
import logging
import time
from typing import Awaitable, Callable

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME = 320  # 20 ms at 16 kHz


class SpeechSegmenter:
    """Cut a PCM16 stream into speech utterances (float32 arrays)."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, max_segment_s: float = 8.0,
                 end_silence_s: float = 0.6, min_speech_s: float = 0.25,
                 preroll_s: float = 0.3, min_threshold: float = 0.008):
        if sample_rate != SAMPLE_RATE:
            raise ValueError("only 16 kHz is supported")
        fps = sample_rate / FRAME
        self.max_frames = max(10, int(max_segment_s * fps))
        self.end_silence_frames = max(1, int(end_silence_s * fps))
        self.min_speech_frames = max(1, int(min_speech_s * fps))
        self.preroll_frames = max(0, int(preroll_s * fps))
        self.min_threshold = min_threshold
        self.reset()

    def reset(self) -> None:
        self._rest = np.zeros(0, dtype=np.float32)  # < 1 frame of leftover samples
        self._odd = b""  # half of a PCM16 sample split across socket frames
        self._preroll: collections.deque = collections.deque(maxlen=self.preroll_frames or 1)
        self._frames: list[np.ndarray] = []   # current utterance
        self._rms: list[float] = []
        self._in_speech = False
        self._speech_frames = 0
        self._silence_frames = 0
        self._noise = self.min_threshold / 3

    @property
    def threshold(self) -> float:
        return max(self.min_threshold, self._noise * 3.0)

    @property
    def buffered_seconds(self) -> float:
        return len(self._frames) * FRAME / SAMPLE_RATE

    def push(self, pcm16: bytes) -> list[np.ndarray]:
        """Feed raw little-endian PCM16 bytes; return finished utterances."""
        # A frame may split a 2-byte sample: carry the odd byte over.
        pcm16 = self._odd + bytes(pcm16)
        self._odd = pcm16[-1:] if len(pcm16) % 2 else b""
        if self._odd:
            pcm16 = pcm16[:-1]
        x = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        if self._rest.size:
            x = np.concatenate([self._rest, x])
        n = (x.size // FRAME) * FRAME
        self._rest = x[n:].copy()
        out: list[np.ndarray] = []
        for i in range(0, n, FRAME):
            seg = self._frame(x[i:i + FRAME])
            if seg is not None:
                out.append(seg)
        return out

    def flush(self) -> np.ndarray | None:
        """End the current utterance now (client asked, or socket closing)."""
        seg = None
        if self._in_speech and self._speech_frames >= self.min_speech_frames and self._frames:
            seg = np.concatenate(self._frames)
        self._end_utterance()
        return seg

    # --- internals ---
    def _frame(self, f: np.ndarray) -> np.ndarray | None:
        rms = float(np.sqrt(np.mean(f * f)))
        loud = rms > self.threshold
        if not self._in_speech:
            if not loud:
                # Track the background level only while nobody is talking.
                self._noise = 0.95 * self._noise + 0.05 * rms
                if self.preroll_frames:
                    self._preroll.append(f)
                return None
            self._in_speech = True
            pre = list(self._preroll) if self.preroll_frames else []
            self._frames = pre + [f]
            self._rms = [0.0] * len(pre) + [rms]
            self._speech_frames, self._silence_frames = 1, 0
            self._preroll.clear()
            return None
        self._frames.append(f)
        self._rms.append(rms)
        if loud:
            self._speech_frames += 1
            self._silence_frames = 0
        else:
            self._silence_frames += 1
        if self._silence_frames >= self.end_silence_frames:
            return self.flush()
        if len(self._frames) >= self.max_frames:
            return self._split_long()
        return None

    def _split_long(self) -> np.ndarray | None:
        """Utterance hit the length cap mid-speech: cut at the quietest
        frame of the last ~1.5 s and carry the remainder over."""
        window = min(len(self._frames) - 1, int(1.5 * SAMPLE_RATE / FRAME))
        start = len(self._frames) - window
        cut = start + int(np.argmin(self._rms[start:])) + 1
        head = self._frames[:cut]
        self._frames, self._rms = self._frames[cut:], self._rms[cut:]
        self._speech_frames = max(1, sum(1 for r in self._rms if r > self.threshold))
        self._silence_frames = 0
        return np.concatenate(head) if head else None

    def _end_utterance(self) -> None:
        self._frames, self._rms = [], []
        self._in_speech = False
        self._speech_frames = self._silence_frames = 0
        self._preroll.clear()


Transcriber = Callable[[np.ndarray, str], Awaitable[dict]]
SendJson = Callable[[dict], Awaitable[None]]


class CaptionSession:
    """One /ws/captions connection after the client chose pcm16."""

    def __init__(self, send_json: SendJson, transcribe: Transcriber, *,
                 language: str = "en", max_segment_s: float = 8.0,
                 max_lag_s: float = 12.0, model_ready: Callable[[], bool] | None = None,
                 load_timeout_s: float = 45.0, clock: Callable[[], float] = time.monotonic):
        self._send = send_json
        self._transcribe = transcribe
        self.language = language
        self.max_lag_s = max_lag_s
        self.load_timeout_s = load_timeout_s
        self._model_ready = model_ready or (lambda: True)
        self._clock = clock
        self._started = clock()
        self._ready_at: float | None = clock() if self._model_ready() else None
        self.segmenter = SpeechSegmenter(max_segment_s=max_segment_s)
        self._queue: asyncio.Queue = asyncio.Queue()
        self._pending_since: collections.deque[float] = collections.deque()  # queued+running
        self._worker = asyncio.create_task(self._run())
        self.failed = False

    # --- input ---
    async def push_audio(self, pcm16: bytes) -> None:
        if self.failed:
            return
        for seg in self.segmenter.push(pcm16):
            self._enqueue(seg)
        await self.check_lag()

    async def flush(self) -> None:
        seg = self.segmenter.flush()
        if seg is not None and not self.failed:
            self._enqueue(seg)

    def reset(self) -> None:
        self.segmenter.reset()

    def _enqueue(self, seg: np.ndarray) -> None:
        now = self._clock()
        self._pending_since.append(now)
        self._queue.put_nowait((seg, now))

    # --- lag / health ---
    def lag_seconds(self) -> float:
        """How long the oldest untranscribed utterance has been waiting
        since the speaker finished it (model-load time excluded)."""
        if not self._pending_since:
            return 0.0
        if self._ready_at is None:
            return 0.0
        return self._clock() - max(self._pending_since[0], self._ready_at)

    async def check_lag(self) -> None:
        if self.failed:
            return
        if self._ready_at is None:
            if self._model_ready():
                self._ready_at = self._clock()
            elif self._clock() - self._started > self.load_timeout_s:
                await self._too_slow(
                    f"The Whisper model did not load within {self.load_timeout_s:.0f} s.")
            return
        lag = self.lag_seconds()
        if lag > self.max_lag_s:
            await self._too_slow(
                f"Whisper on this server is {lag:.0f} s behind the speaker "
                "(not enough CPU for real-time captions).")

    async def _too_slow(self, message: str) -> None:
        self.failed = True
        logger.warning("captions too slow: %s", message)
        await self._send({"type": "error", "code": "too-slow", "message": message})

    # --- worker ---
    async def _run(self) -> None:
        while True:
            seg, queued_at = await self._queue.get()
            if self.failed:
                continue
            if self._ready_at is None and self._model_ready():
                self._ready_at = self._clock()
            try:
                result = await self._transcribe(seg, self.language)
            except Exception as exc:  # model unavailable / decode error
                self.failed = True
                await self._send({"type": "error", "code": "whisper-error", "message": str(exc)})
                continue
            finally:
                if self._pending_since:
                    self._pending_since.popleft()
            if self.failed:
                continue  # already told the client to switch; no late captions
            if self._ready_at is None:
                self._ready_at = self._clock()
            text = (result.get("text") or "").strip()
            if text:
                await self._send({
                    "type": "caption", "text": text,
                    "confidence": result.get("confidence", 0.0), "is_final": True,
                    "engine": result.get("engine", "whisper"),
                    "audio_s": round(len(seg) / SAMPLE_RATE, 2),
                    "latency_s": round(self._clock() - queued_at, 2),
                })
            await self.check_lag()

    async def close(self) -> None:
        self._worker.cancel()
        try:
            await self._worker
        except (asyncio.CancelledError, Exception):
            pass


def parse_config(msg: dict) -> str | None:
    """Validate a client config message; return an error string or None."""
    if msg.get("format") != "pcm16":
        return 'Unsupported audio format; send {"type":"config","format":"pcm16","sample_rate":16000}.'
    if int(msg.get("sample_rate") or 0) != SAMPLE_RATE:
        return "Only 16000 Hz PCM16 mono audio is supported."
    return None
