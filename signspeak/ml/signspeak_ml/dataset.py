"""Dataset, splits and augmentation.

The one thing in this file that will decide whether your reported numbers mean
anything is `leave_one_subject_out`. Use it. A random train/test split on a
sign dataset leaks: multiple recordings of the same person performing the same
sign share idiosyncratic timing, proportions and habits, so a random split puts
near-duplicates on both sides and inflates accuracy by 15-30 points. Every year
a pile of sign-recognition projects report 98% on a random split and then fail
in front of the examiner, who is a new signer. Report signer-independent
accuracy as the headline number and signer-dependent as the secondary one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from .config import CONFIG, LABEL_TO_INDEX, TrainConfig
from .features import FEATURE_DIM, STATIC_DIM, mirror_body_features, resample


@dataclass
class Sample:
    """One recorded sign, with the metadata the robustness slices need.

    If `subject` is missing or everyone is "user1", none of the evaluation in
    this project is meaningful. Collect from at least 6-8 people.
    """

    features: np.ndarray        # (T, FEATURE_DIM)
    label: str
    subject: str
    lighting: str = "normal"    # normal | dim | backlit | harsh
    distance: str = "medium"    # near | medium | far
    background: str = "plain"   # plain | cluttered
    dominant: str = "right"     # right | left
    session: str = "s1"
    source: str = ""            # file path, for tracing a bad sample back

    def meta(self) -> dict:
        return {
            "label": self.label,
            "subject": self.subject,
            "lighting": self.lighting,
            "distance": self.distance,
            "background": self.background,
            "dominant": self.dominant,
            "session": self.session,
            "source": self.source,
        }


def load_dataset(root: str | Path) -> list[Sample]:
    """Load `*.npz` recordings written by capture.py.

    Each npz holds `features` (T, FEATURE_DIM) and a `meta` JSON string.
    """
    root = Path(root)
    samples: list[Sample] = []
    for f in sorted(root.rglob("*.npz")):
        with np.load(f, allow_pickle=False) as z:
            feats = z["features"].astype(np.float32)
            meta = json.loads(str(z["meta"]))
        if feats.ndim != 2 or feats.shape[1] != FEATURE_DIM:
            raise ValueError(
                f"{f}: expected (T, {FEATURE_DIM}) features, got {feats.shape}. "
                "Re-run capture after any change to features.py."
            )
        samples.append(Sample(features=feats, source=str(f), **meta))
    if not samples:
        raise FileNotFoundError(f"no .npz recordings under {root}")
    return samples


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def leave_one_subject_out(samples: list[Sample]) -> list[tuple[list[int], list[int]]]:
    """Yield (train_idx, test_idx) with one subject entirely held out each fold.

    This is the headline protocol. Every number in the report's main table
    should come from averaging these folds, with the standard deviation across
    folds quoted alongside - the spread between subjects is itself a result,
    and usually a large one.
    """
    subjects = sorted({s.subject for s in samples})
    if len(subjects) < 3:
        raise ValueError(
            f"only {len(subjects)} subject(s) in the dataset. Leave-one-subject-out "
            "needs at least 3 to mean anything, and realistically 6-8."
        )
    folds = []
    for held in subjects:
        test = [i for i, s in enumerate(samples) if s.subject == held]
        train = [i for i, s in enumerate(samples) if s.subject != held]
        folds.append((train, test))
    return folds


def subject_stratified_split(
    samples: list[Sample], val_subjects: int = 1, seed: int = 1337
) -> tuple[list[int], list[int]]:
    """Carve a validation set off by subject, for early stopping and for
    fitting the temperature. Never fit calibration on data the model trained
    on - a trained model is overconfident on its own training set by
    construction, so the temperature you'd learn there is meaningless."""
    rng = np.random.default_rng(seed)
    subjects = sorted({s.subject for s in samples})
    rng.shuffle(subjects)
    val = set(subjects[:val_subjects])
    tr = [i for i, s in enumerate(samples) if s.subject not in val]
    va = [i for i, s in enumerate(samples) if s.subject in val]
    return tr, va


def signer_dependent_split(
    samples: list[Sample], test_frac: float = 0.25, seed: int = 1337
) -> tuple[list[int], list[int]]:
    """Random split. Reported only as a *contrast* to the LOSO number, to show
    the reader how large the leak is. Never as the headline."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(samples))
    rng.shuffle(idx)
    cut = int(len(idx) * (1 - test_frac))
    return idx[:cut].tolist(), idx[cut:].tolist()


# ---------------------------------------------------------------------------
# Augmentation - all in body-anchored feature space
# ---------------------------------------------------------------------------
class Augment:
    """Simulates the nuisance variation the feature encoder does not remove.

    Note what is *not* here: brightness, blur, occlusion. Those act on pixels,
    and by the time we have landmarks the pixels are gone. That is a real
    limitation of the landmark-first design - the model never learns to cope
    with a detector that failed, it only ever sees the detector's successes.
    The mitigation is `hand_dropout` below, which simulates tracking loss, plus
    collecting genuinely dim/backlit recordings rather than pretending to.
    """

    def __init__(self, cfg: TrainConfig, rng: np.random.Generator | None = None) -> None:
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(cfg.seed)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        c, r = self.cfg, self.rng
        x = x.copy()
        if r.random() < c.aug_time_warp:
            x = self._time_warp(x)
        if r.random() < c.aug_scale:
            x = x * float(r.normal(1.0, 0.06))
        if r.random() < c.aug_rotate:
            x = self._rotate(x, float(r.normal(0.0, 0.10)))
        if r.random() < c.aug_dropout_hand:
            x = self._hand_dropout(x)
        if r.random() < c.aug_jitter:
            x = x + r.normal(0.0, 0.012, size=x.shape).astype(np.float32)
        return x.astype(np.float32)

    def _time_warp(self, x: np.ndarray) -> np.ndarray:
        """Non-linear speed change: resample through a random monotone curve.
        Uniform speed-up is already removed by fixed-length resampling, so the
        useful augmentation is a *varying* rate within the sign."""
        T = x.shape[0]
        knots = np.sort(self.rng.uniform(0, 1, size=3))
        src = np.concatenate([[0.0], knots, [1.0]])
        dst = np.sort(np.concatenate([[0.0], self.rng.uniform(0, 1, size=3), [1.0]]))
        grid = np.interp(np.linspace(0, 1, T), src, dst) * (T - 1)
        out = np.empty_like(x)
        base = np.arange(T)
        for ch in range(x.shape[1]):
            out[:, ch] = np.interp(grid, base, x[:, ch])
        return out

    def _rotate(self, x: np.ndarray, theta: float) -> np.ndarray:
        """Small in-plane rotation of every 3-vector, simulating a tilted
        camera or a signer leaning. Applies to the static half only; the
        velocity half is rebuilt so the two stay consistent."""
        cos, sin = np.cos(theta), np.sin(theta)
        static = x[:, :STATIC_DIM].copy()
        n3 = (STATIC_DIM // 3) * 3
        v = static[:, :n3].reshape(static.shape[0], -1, 3)
        xs, ys = v[..., 0].copy(), v[..., 1].copy()
        v[..., 0] = cos * xs - sin * ys
        v[..., 1] = sin * xs + cos * ys
        static[:, :n3] = v.reshape(static.shape[0], -1)
        vel = np.zeros_like(static)
        vel[1:] = static[1:] - static[:-1]
        return np.concatenate([static, vel], axis=1)

    def _hand_dropout(self, x: np.ndarray) -> np.ndarray:
        """Zero the non-dominant hand block for a random stretch of frames,
        imitating the detector losing a hand behind the body. Without this the
        model treats an occlusion as strong evidence for whichever class
        happened to occlude most in training."""
        from .features import PER_HAND

        T = x.shape[0]
        lo = int(self.rng.integers(0, max(1, T - 3)))
        hi = min(T, lo + int(self.rng.integers(3, max(4, T // 2))))
        x[lo:hi, PER_HAND : 2 * PER_HAND] = 0.0
        return x


class SignDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        indices: list[int],
        frames: int | None = None,
        augment: Augment | None = None,
    ) -> None:
        self.samples = samples
        self.indices = list(indices)
        self.frames = frames or CONFIG.window.frames
        self.augment = augment

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        s = self.samples[self.indices[i]]
        x = resample(s.features, self.frames)
        if self.augment is not None:
            x = self.augment(x)
        return x.astype(np.float32), LABEL_TO_INDEX[s.label]


def class_weights(samples: list[Sample], indices: list[int], n_classes: int) -> np.ndarray:
    """Inverse-frequency weights. NONE is usually over-represented because
    background is easy to collect; without this the model learns to say NONE."""
    counts = np.zeros(n_classes, dtype=np.float64)
    for i in indices:
        counts[LABEL_TO_INDEX[samples[i].label]] += 1
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (n_classes * counts)
    return (w / w.mean()).astype(np.float32)


def describe(samples: list[Sample]) -> dict:
    """Dataset audit. Run this before training and put it in the report.

    A table of samples-per-class-per-subject makes imbalance and missing cells
    obvious in a way that an overall count never does.
    """
    from collections import Counter, defaultdict

    per_class = Counter(s.label for s in samples)
    per_subject = Counter(s.subject for s in samples)
    grid: dict[str, Counter] = defaultdict(Counter)
    for s in samples:
        grid[s.subject][s.label] += 1
    return {
        "total": len(samples),
        "subjects": len(per_subject),
        "per_class": dict(per_class),
        "per_subject": dict(per_subject),
        "subject_x_class": {k: dict(v) for k, v in grid.items()},
        "lighting": dict(Counter(s.lighting for s in samples)),
        "distance": dict(Counter(s.distance for s in samples)),
        "dominant": dict(Counter(s.dominant for s in samples)),
        "mean_frames": float(np.mean([s.features.shape[0] for s in samples])),
    }
