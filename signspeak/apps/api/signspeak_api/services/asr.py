"""Server-side streaming ASR with faster-whisper.

Read this before wiring it up, because the default plan in the spec has a
latency trap in it.

Whisper is not a streaming model. It consumes a 30-second window and emits a
transcript for the whole thing. Every "streaming Whisper" is a wrapper that
re-runs the model on a growing buffer and diffs the output. That works, and it
is what this file does, but it means:

  * cost grows with utterance length, because you re-transcribe the same audio
    every chunk;
  * the text is unstable - a word can appear, then change, as more context
    arrives, which is disorienting when the person reading it cannot hear the
    audio to check;
  * first-token latency is bounded below by your chunk size.

So the default path in this project is the browser's Web Speech API: genuinely
streaming, free, zero infrastructure, and good enough for the MVP. This server
path exists for three cases the browser cannot cover - Firefox and other
browsers without Web Speech, languages the browser does not support well
(Hindi, for the ISL direction), and any deployment where sending audio to a
third-party recognition service is not acceptable.

Choose deliberately and write down which one produced the numbers in the
report. "We used speech recognition" is not a method section.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import numpy as np

from ..config import settings

SAMPLE_RATE = 16_000


@dataclass
class Hypothesis:
    text: str
    is_final: bool
    started_at_ms: float
    ended_at_ms: float
    utterance_id: str
    avg_logprob: float | None = None

    @property
    def stable_prefix_len(self) -> int:
        return len(self.text)


class EnergyVAD:
    """Voice activity detection by short-term energy with hysteresis.

    Structurally identical to the sign segmenter in `ml/signspeak_ml/segment.py`
    - both answer "when did an utterance start and stop", one for audio and one
    for motion. Worth pointing out in the report: it is the same problem twice,
    and solving it once badly hurts both modalities.

    For production, swap this for Silero VAD. Energy VAD fails in a noisy bank
    branch or a hospital corridor, which are exactly the deployment targets.
    """

    def __init__(
        self,
        frame_ms: int = 30,
        enter_db: float = -38.0,
        exit_db: float = -45.0,
        enter_frames: int = 3,
        exit_frames: int = 12,   # ~360ms of silence ends an utterance
    ) -> None:
        self.frame = int(SAMPLE_RATE * frame_ms / 1000)
        self.enter_db, self.exit_db = enter_db, exit_db
        self.enter_frames, self.exit_frames = enter_frames, exit_frames
        self.active = False
        self._above = self._below = 0

    def push(self, pcm: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2)) + 1e-9)
        db = 20 * np.log10(rms)
        if not self.active:
            self._above = self._above + 1 if db > self.enter_db else 0
            if self._above >= self.enter_frames:
                self.active, self._below = True, 0
        else:
            self._below = self._below + 1 if db < self.exit_db else 0
            if self._below >= self.exit_frames:
                self.active, self._above = False, 0
        return self.active


@dataclass
class StreamState:
    buffer: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    utterance_id: str = ""
    started_at_ms: float = 0.0
    last_text: str = ""


class WhisperStreamer:
    """Sliding-window pseudo-streaming over faster-whisper.

    Transcribes the active utterance every `interval_ms`, emits the result as a
    partial, and emits a final when VAD says the utterance ended. The model is
    loaded once and guarded by a semaphore, because two concurrent calls on a
    CPU-bound int8 model will each take twice as long rather than running in
    parallel - a detail that turns "works with one user" into "unusable with
    three".
    """

    _model = None
    _load_lock = asyncio.Lock()

    def __init__(self, interval_ms: int = 600, max_concurrency: int = 2) -> None:
        self.interval_ms = interval_ms
        self.vad = EnergyVAD()
        self.state = StreamState()
        self._sem = asyncio.Semaphore(max_concurrency)
        self._last_run = 0.0

    @classmethod
    async def model(cls):
        if cls._model is None:
            async with cls._load_lock:
                if cls._model is None:
                    from faster_whisper import WhisperModel

                    cls._model = await asyncio.to_thread(
                        WhisperModel,
                        settings.whisper_model,
                        device=settings.whisper_device,
                        compute_type=settings.whisper_compute_type,
                    )
        return cls._model

    async def feed(self, pcm: np.ndarray, now_ms: float) -> AsyncIterator[Hypothesis]:
        """Push one audio chunk (float32 mono @16kHz). Yields 0..n hypotheses."""
        was_active = self.vad.active
        active = self.vad.push(pcm)

        if active and not was_active:
            self.state = StreamState(
                buffer=np.zeros(0, dtype=np.float32),
                utterance_id=f"u{int(now_ms)}",
                started_at_ms=now_ms,
            )

        if active:
            self.state.buffer = np.concatenate([self.state.buffer, pcm])
            if (now_ms - self._last_run) >= self.interval_ms:
                self._last_run = now_ms
                text = await self._transcribe(self.state.buffer)
                if text and text != self.state.last_text:
                    self.state.last_text = text
                    yield Hypothesis(text, False, self.state.started_at_ms, now_ms,
                                     self.state.utterance_id)

        elif was_active and not active:
            text = await self._transcribe(self.state.buffer)
            if text:
                yield Hypothesis(text, True, self.state.started_at_ms, now_ms,
                                 self.state.utterance_id)
            self.state = StreamState()
            self._last_run = 0.0

    async def _transcribe(self, audio: np.ndarray) -> str:
        if audio.size < SAMPLE_RATE * 0.25:      # < 250ms: nothing useful in there
            return ""
        model = await self.model()
        async with self._sem:
            segments, _ = await asyncio.to_thread(
                model.transcribe,
                audio,
                language="en",
                beam_size=1,               # greedy: ~2x faster, negligible WER cost here
                vad_filter=False,          # our VAD already cut the utterance
                condition_on_previous_text=False,  # stops hallucinated repetition
            )
            return " ".join(s.text.strip() for s in segments).strip()


def pcm16_to_float32(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def now_ms() -> float:
    return time.monotonic() * 1000.0
