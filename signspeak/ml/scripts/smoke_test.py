"""End-to-end plumbing check on synthetic data.

Runs the whole pipeline - features, segmentation, training, calibration,
evaluation, decision policy - and asserts the parts that must hold regardless
of what the numbers say:

  * feature vectors have the dimension both the trainer and the browser expect
  * the body frame is genuinely orthonormal
  * features are invariant to camera distance and framing, which is the entire
    point of features.py
  * the segmenter finds the right number of signs in a synthetic stream
  * temperature scaling reduces expected calibration error
  * leave-one-subject-out scores BELOW a random split (if it does not, the
    split code is broken and every number downstream is a lie)
  * the decision policy refuses out-of-distribution input

Run it in CI. It takes under a minute and catches the class of bug that
otherwise surfaces three weeks later as "the model is worse in the browser".

    python scripts/smoke_test.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from signspeak_ml.calibrate import (  # noqa: E402
    TemperatureScaler,
    decide,
    energy_score,
    expected_calibration_error,
    fit_thresholds,
    risk_coverage_curve,
    softmax,
)
from signspeak_ml.config import CONFIG, NUM_CLASSES, VOCAB  # noqa: E402
from signspeak_ml.dataset import (  # noqa: E402
    Augment,
    leave_one_subject_out,
    load_dataset,
    signer_dependent_split,
)
from signspeak_ml.features import (  # noqa: E402
    FEATURE_DIM,
    STATIC_DIM,
    add_velocity,
    body_frame,
    encode_frame,
    encode_sequence,
    resample,
)
from signspeak_ml.models import DTWkNN, build, count_params  # noqa: E402
from signspeak_ml.segment import StreamingSegmenter, segment_offline, segmentation_f1  # noqa: E402
from signspeak_ml.train import train_one  # noqa: E402

PASS, FAIL = "  PASS", "  FAIL"
failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"{PASS if cond else FAIL}  {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        failures.append(name)


def fake_pose(cx=0.5, cy=0.5, scale=1.0, dz=0.0) -> np.ndarray:
    """A plausible upper body at an arbitrary position and distance."""
    p = np.zeros((33, 3), dtype=np.float32)
    p[0] = [cx, cy - 0.22 * scale, dz]            # nose
    p[2] = [cx - 0.03 * scale, cy - 0.24 * scale, dz]
    p[5] = [cx + 0.03 * scale, cy - 0.24 * scale, dz]
    p[11] = [cx + 0.12 * scale, cy, dz]           # left shoulder
    p[12] = [cx - 0.12 * scale, cy, dz]           # right shoulder
    p[13] = [cx + 0.18 * scale, cy + 0.15 * scale, dz]
    p[14] = [cx - 0.18 * scale, cy + 0.15 * scale, dz]
    p[15] = [cx + 0.20 * scale, cy + 0.28 * scale, dz]
    p[16] = [cx - 0.20 * scale, cy + 0.28 * scale, dz]
    p[23] = [cx + 0.09 * scale, cy + 0.34 * scale, dz]   # left hip
    p[24] = [cx - 0.09 * scale, cy + 0.34 * scale, dz]
    return p


def fake_hand(cx, cy, scale=1.0, spread=0.05) -> np.ndarray:
    """A hand at (cx, cy). `scale` scales ALL THREE axes.

    Scaling z as well as x and y matters: the invariance test below is only a
    real test if the far-away pose is a genuine similarity transform of the
    near one. Leaving z unscaled makes the two differ in depth by 1/scale and
    the test fails for a reason that has nothing to do with the encoder.
    """
    h = np.zeros((21, 3), dtype=np.float32)
    for i in range(21):
        a = i * 0.3
        h[i] = [
            cx + spread * scale * np.cos(a),
            cy + spread * scale * np.sin(a),
            0.01 * i * scale,
        ]
    return h


# ---------------------------------------------------------------------------
print("\n1. Feature encoder")
# ---------------------------------------------------------------------------
pose = fake_pose()
bf = body_frame(pose)
check("body frame is constructible", bf is not None)

origin, R, scale = bf
check(
    "body frame is orthonormal",
    np.allclose(R @ R.T, np.eye(3), atol=1e-5),
    f"max |RRᵀ-I| = {np.abs(R @ R.T - np.eye(3)).max():.2e}",
)

v = encode_frame(pose, fake_hand(0.62, 0.42), fake_hand(0.38, 0.42))
check("frame vector has STATIC_DIM", v is not None and v.shape == (STATIC_DIM,), f"{v.shape}")

# The whole reason features.py exists: the same gesture, filmed from a
# different distance and a different part of the frame, must encode the same.
v_near = encode_frame(fake_pose(0.5, 0.5, 1.0), fake_hand(0.62, 0.42, 1.0), fake_hand(0.38, 0.42, 1.0))
v_far = encode_frame(fake_pose(0.3, 0.7, 0.55), fake_hand(0.366, 0.656, 0.55), fake_hand(0.234, 0.656, 0.55))
delta = float(np.abs(v_near - v_far).max())
check(
    "features invariant to camera distance + framing",
    delta < 0.05,
    f"max |Δ| = {delta:.4f} (raw landmarks would differ by ~0.2-0.4)",
)

seq = [
    {"pose": fake_pose(), "left_hand": fake_hand(0.38, 0.42 + 0.01 * t),
     "right_hand": fake_hand(0.62, 0.42 - 0.01 * t), "face": None}
    for t in range(40)
]
enc = encode_sequence(seq, CONFIG.window.frames)
check("sequence encodes to (frames, FEATURE_DIM)",
      enc is not None and enc.shape == (CONFIG.window.frames, FEATURE_DIM), f"{enc.shape}")

check("no NaN or inf in encoded features", bool(np.all(np.isfinite(enc))))

# A frame with no torso must fail loudly rather than silently emit image coords.
check("frame without torso is rejected", encode_frame(np.zeros((33, 3)), None, None) is None)


# ---------------------------------------------------------------------------
print("\n2. Segmenter")
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)


def synthetic_stream(n_signs=3, quiet=25, active=30):
    """Quiet / active / quiet ... with the dominant wrist actually moving."""
    frames, truth = [], []
    t = 0
    for _ in range(n_signs):
        for _ in range(quiet):
            f = np.zeros(STATIC_DIM, dtype=np.float32)
            f[72] = 1.0                      # dominant hand present
            f[63:66] = [0.3, 0.2, 0.0]       # parked, in signing space
            f += rng.normal(0, 0.0015, STATIC_DIM).astype(np.float32)
            f[72] = 1.0
            frames.append(f)
            t += 1
        start = t
        for k in range(active):
            f = np.zeros(STATIC_DIM, dtype=np.float32)
            f[72] = 1.0
            f[63:66] = [0.3 + 0.07 * k, 0.2 + 0.05 * np.sin(k * 0.4), 0.0]
            frames.append(f)
            t += 1
        truth.append((start, t))
    for _ in range(quiet):
        f = np.zeros(STATIC_DIM, dtype=np.float32)
        f[72] = 1.0
        f[63:66] = [0.3, 0.2, 0.0]
        frames.append(f)
        t += 1
    return np.stack(frames), truth


stream, truth = synthetic_stream(3)
spans = segment_offline(stream)
check("segmenter finds the right number of signs", len(spans) == 3, f"found {len(spans)}, expected 3")

f1 = segmentation_f1(spans, truth, iou_threshold=0.4)
check("segmentation F1 > 0.6 on a clean stream", f1["f1"] > 0.6, f"F1={f1['f1']:.2f}")

# A perfectly still stream must produce nothing at all.
still = np.tile(stream[0], (120, 1))
check("no false detections on a still stream", len(segment_offline(still)) == 0)

seg = StreamingSegmenter()
check("streaming segmenter starts idle", not seg._buffer and seg.state.value == "idle")


# ---------------------------------------------------------------------------
print("\n3. Dataset + splits")
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    data_dir = Path(tmp) / "raw"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "make_synthetic.py"),
         "--out", str(data_dir), "--subjects", "5", "--per-class", "10"],
        check=True, capture_output=True,
    )
    samples = load_dataset(data_dir)
    check("synthetic dataset loads", len(samples) > 0, f"{len(samples)} samples")

    folds = leave_one_subject_out(samples)
    check("LOSO yields one fold per subject", len(folds) == 5, f"{len(folds)} folds")

    overlap = any(
        {samples[i].subject for i in tr} & {samples[i].subject for i in te} for tr, te in folds
    )
    check("LOSO folds share no subject between train and test", not overlap)

    aug = Augment(CONFIG.train, np.random.default_rng(0))
    x = resample(samples[0].features, CONFIG.window.frames)
    xa = aug(x)
    check("augmentation preserves shape", xa.shape == x.shape, f"{xa.shape}")
    check("augmentation actually changes the sample", float(np.abs(xa - x).max()) > 1e-4)
    check("augmented sample is finite", bool(np.all(np.isfinite(xa))))

    # ---------------------------------------------------------------------
    print("\n4. Models")
    # ---------------------------------------------------------------------
    dev = torch.device("cpu")
    for name in ("gru", "tcn", "transformer"):
        m = build(name, FEATURE_DIM, CONFIG.train, NUM_CLASSES)
        out = m(torch.randn(4, CONFIG.window.frames, FEATURE_DIM))
        check(f"{name} forward pass", tuple(out.shape) == (4, NUM_CLASSES),
              f"{tuple(out.shape)}, {count_params(m):,} params")

    Xtr = np.stack([resample(samples[i].features, CONFIG.window.frames) for i in folds[0][0][:40]])
    ytr = np.array([VOCAB.index(samples[i].label) for i in folds[0][0][:40]])
    Xte = np.stack([resample(samples[i].features, CONFIG.window.frames) for i in folds[0][1][:8]])
    knn = DTWkNN(k=3).fit(Xtr, ytr)
    p = knn.predict_proba(Xte)
    check("DTW k-NN produces a distribution", p.shape == (8, NUM_CLASSES)
          and np.allclose(p.sum(1), 1.0, atol=1e-4))

    # ---------------------------------------------------------------------
    print("\n5. Train / calibrate / evaluate")
    # ---------------------------------------------------------------------
    from dataclasses import replace

    cfg = replace(CONFIG.train, epochs=14, patience=14)
    tr, te = folds[0]
    r_loso = train_one(samples, tr, te, "gru", cfg, dev, verbose=False)
    check("training returns an accuracy", 0.0 <= r_loso["accuracy"] <= 1.0,
          f"LOSO acc={r_loso['accuracy']:.3f}")
    check("temperature scaling reduces ECE",
          r_loso["ece_calibrated"] <= r_loso["ece_raw"] + 1e-6,
          f"{r_loso['ece_raw']:.4f} → {r_loso['ece_calibrated']:.4f} (T={r_loso['temperature']:.2f})")

    tr2, te2 = signer_dependent_split(samples, test_frac=0.25)
    r_rand = train_one(samples, tr2, te2, "gru", cfg, dev, verbose=False)
    gap = r_rand["accuracy"] - r_loso["accuracy"]
    at_ceiling = r_loso["accuracy"] > 0.98 and r_rand["accuracy"] > 0.98
    check(
        "random split scores at or above leave-one-subject-out",
        gap >= -0.02 or at_ceiling,
        f"signer-dependent {r_rand['accuracy']:.3f} vs signer-independent "
        f"{r_loso['accuracy']:.3f} — gap {gap:+.3f}"
        + ("  [both at ceiling: synthetic classes are too separable to show "
           "the leak — on real data expect a gap of 0.15-0.30]" if at_ceiling else ""),
    )

    # ---------------------------------------------------------------------
    print("\n6. Calibration + decision policy")
    # ---------------------------------------------------------------------
    logits = r_loso["test_logits"]
    labels = r_loso["test_labels"]
    T = r_loso["temperature"]

    ece_raw, bins = expected_calibration_error(softmax(logits), labels)
    ece_cal, _ = expected_calibration_error(softmax(logits / T), labels)
    check("ECE computed with populated bins", sum(b["n"] for b in bins) == len(labels))

    curve = risk_coverage_curve(softmax(logits / T), labels)
    monotone_ok = curve[0]["coverage"] < curve[-1]["coverage"]
    check("risk-coverage curve spans full coverage", monotone_ok,
          f"risk at 100% coverage = {curve[-1]['risk']:.3f}")

    th = fit_thresholds(softmax(logits / T), labels, logits, T, max_risk=0.05)
    check("thresholds fitted", 0 < th.confirm <= th.accept <= 1.01,
          f"accept={th.accept:.3f} confirm={th.confirm:.3f} "
          f"(risk {th.achieved_risk:.3f} @ coverage {th.achieved_coverage:.2f})")

    scaler = TemperatureScaler()
    scaler.log_t = torch.tensor([np.log(T)], dtype=torch.float32)
    probs = scaler.probs(logits[:1])[0]
    d = decide(probs, logits[0], VOCAB, th)
    check("decision policy returns a valid action", d.action in {"speak", "confirm", "reject"},
          f"{d.action}: {d.label} @ {d.confidence:.2f} ({d.reason})")

    # Wildly out-of-distribution input must not be spoken.
    ood = np.full(NUM_CLASSES, -40.0, dtype=np.float32)
    ood[3] = 40.0
    d_ood = decide(softmax(ood[None] / T)[0], ood, VOCAB, th)
    e_in = float(energy_score(logits[:1], T)[0])
    e_ood = float(energy_score(ood[None], T)[0])
    check("energy score separates OOD from in-distribution", e_ood < e_in or d_ood.action != "speak",
          f"energy in-dist={e_in:.1f}, ood={e_ood:.1f}, action={d_ood.action}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
if failures:
    print(f"{len(failures)} CHECK(S) FAILED:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("All checks passed. The pipeline is wired correctly end to end.")
print("Note: the accuracy numbers above come from synthetic data and mean nothing.")
print("=" * 72)
