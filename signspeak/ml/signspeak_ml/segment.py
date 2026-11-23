"""Streaming sign segmentation.

The original SignSpeak spec jumps from "landmarks" straight to "temporal model
-> recognized sign". That works in a lab where you press record, sign once, and
press stop. In a live call nobody presses anything: the camera produces an
unbroken stream containing signs, gesticulation, scratching, reaching for a
glass of water, and long stretches of nothing. Something has to decide *when* a
sign happened. That decision is this file, and it is the hardest unsolved piece
of the MVP - harder than the classifier.

The structure is borrowed wholesale from voice activity detection, which solves
the same shape of problem for audio:

    per-frame energy -> hysteresis state machine -> bounded segments

Hysteresis (enter high, exit low) matters. A single threshold produces chatter:
energy dithers across the line mid-sign and one sign is emitted as three.

Known limits, stated plainly because the report needs them:
  * Co-articulation. In continuous signing, signs run into each other and there
    is no low-energy boundary between them. This segmenter will merge them.
    That is acceptable for an isolated-sign MVP and fatal for continuous
    recognition, which needs CTC or a transducer instead of segment+classify.
  * End-of-sign lag. We only know a sign ended after `exit_frames` of quiet,
    which at 25fps is ~280ms of pure latency on every utterance. This, not
    model inference, dominates the end-to-end budget. See
    docs/architecture.md "Latency budget".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .config import CONFIG, SegmenterConfig, WindowConfig
from .features import PER_HAND, PER_HAND_SHAPE, POSE_KEEP, POSE_KEEP_ORDER


class State(Enum):
    IDLE = "idle"
    MAYBE_ACTIVE = "maybe_active"
    ACTIVE = "active"
    MAYBE_DONE = "maybe_done"


@dataclass
class Segment:
    start: int
    end: int                  # exclusive
    frames: np.ndarray        # (T, STATIC_DIM)
    peak_energy: float
    truncated: bool = False   # hit max_segment_frames rather than ending naturally


def hands_in_signing_space(static_frame: np.ndarray) -> bool:
    """True if the dominant hand is above the hip line in the body frame.

    Body-frame y runs shoulders -> hips, so y ~ 0 is shoulder height and y ~ 1
    is hip height. Hands resting in the lap sit well past 1. A generous cut at
    1.15 removes most resting-hand motion without clipping low signs.
    """
    present = static_frame[PER_HAND - 1]
    if present < 0.5:
        return False
    wrist_y = static_frame[PER_HAND_SHAPE + 1]
    return bool(wrist_y < 1.15)


class StreamingSegmenter:
    """Frame-at-a-time segmenter. Feed it, and it hands back finished segments.

    Mirrored in `apps/web/lib/segmenter.ts`; keep the state machines identical.
    """

    def __init__(
        self,
        cfg: SegmenterConfig | None = None,
        window: WindowConfig | None = None,
    ) -> None:
        self.cfg = cfg or CONFIG.segmenter
        self.window = window or CONFIG.window
        self.reset()

    def reset(self) -> None:
        self.state = State.IDLE
        self._buffer: list[np.ndarray] = []      # rolling context + active frames
        self._energies: list[float] = []
        self._counter = 0
        self._prev_wrist: np.ndarray | None = None
        self._active_start: int | None = None
        self._frame_index = 0
        self._peak = 0.0

    # -- energy ------------------------------------------------------------
    def _energy(self, static_frame: np.ndarray) -> float:
        wrist = static_frame[PER_HAND_SHAPE : PER_HAND_SHAPE + 3]
        present = static_frame[PER_HAND - 1] > 0.5
        if not present:
            self._prev_wrist = None
            return 0.0
        if self._prev_wrist is None:
            self._prev_wrist = wrist.copy()
            return 0.0
        e = float(np.linalg.norm(wrist - self._prev_wrist))
        self._prev_wrist = wrist.copy()
        return e

    # -- main loop ---------------------------------------------------------
    def push(self, static_frame: np.ndarray | None) -> Segment | None:
        """Feed one encoded frame. Returns a Segment on the frame a sign ends.

        `None` means the frame was unusable (no torso). We treat that as
        silence rather than dropping it, so a tracking failure mid-sign closes
        the segment instead of stitching across a gap of unknown length.
        """
        idx = self._frame_index
        self._frame_index += 1

        if static_frame is None:
            return self._on_quiet(idx, np.zeros(1, dtype=np.float32), force=True)

        e = self._energy(static_frame)
        in_space = (
            hands_in_signing_space(static_frame)
            if self.cfg.require_hands_in_signing_space
            else True
        )
        loud = e > self.cfg.enter_threshold and in_space
        quiet = e < self.cfg.exit_threshold or not in_space

        # Always keep a short pre-roll so a segment can include the frames just
        # before motion crossed the threshold - the start of a sign is low
        # energy by definition.
        self._buffer.append(static_frame)
        self._energies.append(e)

        # Trimming the rolling buffer shifts every index in it. `_active_start`
        # is an index INTO this buffer, so it has to move with it - otherwise
        # it silently points at the wrong frame once a call runs longer than
        # `keep` frames, and segments after that are the wrong length or lost
        # entirely. This bug does not appear in short tests; it appears about
        # four seconds into a real conversation.
        keep = self.window.max_segment_frames + self.cfg.pad_frames + 4
        if len(self._buffer) > keep:
            drop = len(self._buffer) - keep
            del self._buffer[:drop]
            del self._energies[:drop]
            if self._active_start is not None:
                self._active_start = max(0, self._active_start - drop)

        if self.state is State.IDLE:
            if loud:
                self.state = State.MAYBE_ACTIVE
                self._counter = 1
                self._active_start = max(0, len(self._buffer) - 1 - self.cfg.pad_frames)
                self._peak = e
            return None

        if self.state is State.MAYBE_ACTIVE:
            if loud:
                self._counter += 1
                self._peak = max(self._peak, e)
                if self._counter >= self.cfg.enter_frames:
                    self.state = State.ACTIVE
            else:
                self.state = State.IDLE
                self._counter = 0
                self._active_start = None
            return None

        if self.state is State.ACTIVE:
            self._peak = max(self._peak, e)
            if quiet:
                self.state = State.MAYBE_DONE
                self._counter = 1
            return self._maybe_truncate(idx)

        # MAYBE_DONE
        if loud:
            self.state = State.ACTIVE
            self._counter = 0
            self._peak = max(self._peak, e)
            return self._maybe_truncate(idx)

        self._counter += 1
        if self._counter >= self.cfg.exit_frames:
            return self._close(idx, truncated=False)
        return self._maybe_truncate(idx)

    def _maybe_truncate(self, idx: int) -> Segment | None:
        if self._active_start is None:
            return None
        length = len(self._buffer) - self._active_start
        if length >= self.window.max_segment_frames:
            return self._close(idx, truncated=True)
        return None

    def _on_quiet(self, idx: int, _dummy, force: bool) -> Segment | None:
        if self.state in (State.ACTIVE, State.MAYBE_DONE):
            return self._close(idx, truncated=False)
        self.state = State.IDLE
        self._counter = 0
        self._active_start = None
        self._prev_wrist = None
        return None

    def _close(self, idx: int, truncated: bool) -> Segment | None:
        start = self._active_start
        self.state = State.IDLE
        self._counter = 0
        self._active_start = None
        if start is None:
            return None

        # Drop the trailing quiet frames used to detect the end, keeping pad.
        tail = 0 if truncated else max(0, self.cfg.exit_frames - self.cfg.pad_frames)
        end = max(start + 1, len(self._buffer) - tail)
        frames = np.stack(self._buffer[start:end], axis=0)

        # Keep only a short pre-roll for the next sign. Without this the buffer
        # accumulates the whole call and every index arithmetic above gets
        # progressively more fragile.
        tail_keep = self.cfg.pad_frames + 2
        if len(self._buffer) > tail_keep:
            del self._buffer[:-tail_keep]
            del self._energies[:-tail_keep]

        if frames.shape[0] < self.window.min_segment_frames:
            return None  # too short to be a sign; discard silently

        seg = Segment(
            start=idx - (end - start),
            end=idx,
            frames=frames,
            peak_energy=self._peak,
            truncated=truncated,
        )
        self._peak = 0.0
        return seg


# ---------------------------------------------------------------------------
# Offline segmentation, for evaluating the segmenter itself
# ---------------------------------------------------------------------------
def segment_offline(
    static_seq: np.ndarray, cfg: SegmenterConfig | None = None
) -> list[tuple[int, int]]:
    """Run the same state machine over a whole recording. Returns [(start, end)]."""
    seg = StreamingSegmenter(cfg)
    spans: list[tuple[int, int]] = []
    for i in range(static_seq.shape[0]):
        out = seg.push(static_seq[i])
        if out is not None:
            spans.append((out.start, out.end))
    final = seg.push(None)
    if final is not None:
        spans.append((final.start, final.end))
    return spans


def segmentation_f1(
    predicted: list[tuple[int, int]],
    truth: list[tuple[int, int]],
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Detection-style F1 over temporal spans, matched greedily by IoU.

    This is the metric to report for the segmenter. Classification accuracy on
    pre-cut clips says nothing about whether the system works on a live stream;
    this does.
    """

    def iou(a: tuple[int, int], b: tuple[int, int]) -> float:
        inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
        union = max(a[1], b[1]) - min(a[0], b[0])
        return inter / union if union > 0 else 0.0

    unmatched = list(range(len(truth)))
    tp = 0
    for p in predicted:
        best, best_iou = -1, 0.0
        for ti in unmatched:
            v = iou(p, truth[ti])
            if v > best_iou:
                best, best_iou = ti, v
        if best >= 0 and best_iou >= iou_threshold:
            tp += 1
            unmatched.remove(best)

    fp = len(predicted) - tp
    fn = len(truth) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
