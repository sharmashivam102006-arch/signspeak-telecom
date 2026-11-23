"""Generate a synthetic dataset so the whole pipeline runs before you own a camera.

This exists for one reason: you should find out that train.py crashes, or that
your ONNX export diverges, *now*, not on the evening you finally have eight
people in a room. It is a plumbing test, not a science experiment - the numbers
it produces are meaningless and must never appear in the report.

Each synthetic "sign" is a distinct smooth trajectory of a fake wrist through
the body frame, with per-subject bias (people differ), per-sample noise, and a
NONE class of random drift. Subject bias is deliberately large enough that
leave-one-subject-out scores clearly below a random split - which is exactly
the effect the real data will show, and a good way to confirm the split code
does what it claims.

    python scripts/make_synthetic.py --subjects 6 --per-class 24
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from signspeak_ml.config import VOCAB  # noqa: E402
from signspeak_ml.features import (  # noqa: E402
    FEATURE_DIM,
    PER_HAND,
    PER_HAND_SHAPE,
    STATIC_DIM,
    add_velocity,
)


def trajectory(label_idx: int, T: int, rng: np.random.Generator) -> np.ndarray:
    """A per-class 3D path for the dominant wrist, plus a handshape signature."""
    t = np.linspace(0, 1, T)
    phase = label_idx * 0.9
    x = 0.45 * np.sin(2 * np.pi * (0.6 + 0.35 * label_idx) * t + phase)
    y = -0.35 + 0.30 * np.cos(2 * np.pi * (0.4 + 0.25 * label_idx) * t + phase)
    z = 0.18 * np.sin(2 * np.pi * (0.9 + 0.2 * label_idx) * t)
    return np.stack([x, y, z], axis=1)


def make_sample(label: str, subject_bias: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    T = int(rng.integers(20, 55))
    static = np.zeros((T, STATIC_DIM), dtype=np.float32)

    if label == "NONE":
        # Random low-frequency drift: movement with no class structure.
        path = np.cumsum(rng.normal(0, 0.05, size=(T, 3)), axis=0) * 0.3
        shape_sig = rng.normal(0, 0.4, size=PER_HAND_SHAPE)
    else:
        li = VOCAB.index(label)
        path = trajectory(li, T, rng)
        shape_sig = np.sin(np.arange(PER_HAND_SHAPE) * (0.3 + 0.11 * li)) * 0.8

    # dominant hand: shape signature, then wrist path, orientation, presence
    static[:, :PER_HAND_SHAPE] = shape_sig + rng.normal(0, 0.06, size=PER_HAND_SHAPE)
    static[:, PER_HAND_SHAPE : PER_HAND_SHAPE + 3] = path + subject_bias[None, :3]
    static[:, PER_HAND_SHAPE + 3 : PER_HAND_SHAPE + 9] = rng.normal(0, 0.15, size=6)
    static[:, PER_HAND - 1] = 1.0

    # non-dominant hand present for roughly half the vocabulary
    if label not in ("NONE",) and VOCAB.index(label) % 2 == 0:
        static[:, PER_HAND : PER_HAND + PER_HAND_SHAPE] = shape_sig * 0.6
        static[:, 2 * PER_HAND - 1] = 1.0

    # pose + face blocks: subject-specific constants (body proportions)
    static[:, 2 * PER_HAND :] = subject_bias[3:][None, :] * 0.5

    static += rng.normal(0, 0.03, size=static.shape).astype(np.float32)
    return add_velocity(static.astype(np.float32))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--subjects", type=int, default=6)
    ap.add_argument("--per-class", type=int, default=24)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    root = Path(args.out)
    lightings = ["normal", "normal", "dim", "backlit"]
    distances = ["medium", "medium", "near", "far"]

    n = 0
    for s in range(args.subjects):
        subject = f"synthetic{s + 1:02d}"
        bias = rng.normal(0, 0.28, size=STATIC_DIM - 2 * PER_HAND + 3)
        dominant = "left" if s % 5 == 4 else "right"
        d = root / subject / "synthetic"
        d.mkdir(parents=True, exist_ok=True)
        for label in VOCAB:
            count = args.per_class if label != "NONE" else int(args.per_class * 1.5)
            for _ in range(count):
                feats = make_sample(label, bias, rng)
                assert feats.shape[1] == FEATURE_DIM
                np.savez_compressed(
                    d / f"{label}_{uuid.uuid4().hex[:8]}.npz",
                    features=feats,
                    meta=json.dumps(
                        {
                            "label": label,
                            "subject": subject,
                            "lighting": str(rng.choice(lightings)),
                            "distance": str(rng.choice(distances)),
                            "background": "plain",
                            "dominant": dominant,
                            "session": "synthetic",
                        }
                    ),
                )
                n += 1

    print(f"wrote {n} synthetic samples across {args.subjects} subjects -> {root}")
    print("These numbers are plumbing, not results. Do not put them in the report.")


if __name__ == "__main__":
    main()
