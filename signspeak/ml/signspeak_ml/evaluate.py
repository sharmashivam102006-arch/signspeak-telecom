"""Evaluation harness.

Produces everything Phase 7 of the roadmap asks for, and the things it should
have asked for: per-class metrics, a confusion matrix, calibration, the
risk-coverage curve, robustness slices, latency, and a baseline comparison.

    python -m signspeak_ml.evaluate --data data/raw --checkpoint artifacts/gru.pt
    python -m signspeak_ml.evaluate --data data/raw --baseline    # DTW k-NN only

Writes `artifacts/evaluation.json` and a readable `artifacts/report.md`.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .calibrate import (
    TemperatureScaler,
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    risk_coverage_curve,
    softmax,
)
from .config import CONFIG, LABEL_TO_INDEX, NONE_INDEX, NUM_CLASSES, VOCAB
from .dataset import SignDataset, leave_one_subject_out, load_dataset
from .features import FEATURE_DIM, resample
from .models import DTWkNN, build


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n: int = NUM_CLASSES) -> np.ndarray:
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def per_class_metrics(cm: np.ndarray, vocab: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for i, name in enumerate(vocab):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        out[name] = {
            "precision": p,
            "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "support": int(cm[i, :].sum()),
        }
    return out


def macro_f1(per_class: dict[str, dict], exclude: tuple[str, ...] = ("NONE",)) -> float:
    vals = [m["f1"] for k, m in per_class.items() if k not in exclude and m["support"] > 0]
    return float(np.mean(vals)) if vals else 0.0


def top_confusions(cm: np.ndarray, vocab: list[str], k: int = 5) -> list[dict]:
    """The pairs the model actually mixes up. Worth a paragraph each in the
    report - "HELP and EMERGENCY share a handshape and differ mainly in
    movement amplitude" is analysis; a confusion matrix image is not."""
    pairs = [
        {"true": vocab[i], "pred": vocab[j], "count": int(cm[i, j])}
        for i in range(len(vocab))
        for j in range(len(vocab))
        if i != j and cm[i, j] > 0
    ]
    return sorted(pairs, key=lambda d: -d["count"])[:k]


# ---------------------------------------------------------------------------
# Robustness slices
# ---------------------------------------------------------------------------
def slice_metrics(samples, indices, y_true, y_pred, attribute: str) -> dict[str, dict]:
    """Accuracy broken down by a recording condition.

    The headline number hides the failure mode. A system at 92% overall but 61%
    in dim light is not a 92% system for anyone who takes a call in the
    evening, and that sentence belongs in the report.
    """
    buckets: dict[str, list[int]] = defaultdict(list)
    for pos, idx in enumerate(indices):
        buckets[getattr(samples[idx], attribute)].append(pos)
    out = {}
    for key, pos in sorted(buckets.items()):
        pos_arr = np.array(pos)
        out[key] = {
            "n": len(pos),
            "accuracy": float((y_pred[pos_arr] == y_true[pos_arr]).mean()),
        }
    return out


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------
def measure_latency(model, frames: int, dim: int, runs: int = 200, device="cpu") -> dict:
    """Model inference latency only.

    This is the smallest term in the end-to-end budget and the one everyone
    reports. Measure it, then put it next to the segmentation lag in
    docs/architecture.md so the reader sees which one actually matters.
    """
    model.eval()
    x = torch.randn(1, frames, dim, device=device)
    with torch.no_grad():
        for _ in range(20):
            model(x)
        ts = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            ts.append((time.perf_counter() - t0) * 1000)
    a = np.array(ts)
    return {
        "mean_ms": float(a.mean()),
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def write_report(path: Path, r: dict) -> None:
    L = ["# SignSpeak - sign recognition evaluation", ""]
    L += [
        f"Protocol: **{r['protocol']}** | folds: {r['n_folds']} | "
        f"subjects: {r['n_subjects']} | samples: {r['n_samples']}",
        "",
        "## Headline",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| Accuracy (mean over folds) | {r['accuracy_mean']:.3f} +/- {r['accuracy_std']:.3f} |",
        f"| Accuracy (worst subject) | {r['accuracy_min']:.3f} |",
        f"| Macro F1 (signs only) | {r['macro_f1']:.3f} |",
        f"| ECE before calibration | {r['ece_raw']:.3f} |",
        f"| ECE after calibration | {r['ece_calibrated']:.3f} |",
        f"| Brier score | {r['brier']:.3f} |",
        f"| Inference p95 | {r['latency']['p95_ms']:.2f} ms |",
        "",
        "> The worst-subject number is the one to quote when someone asks whether it "
        "works. The mean tells you about your dataset; the minimum tells you about "
        "the next person who uses it.",
        "",
        "## Per-class",
        "",
        "| sign | precision | recall | F1 | support |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, m in r["per_class"].items():
        L.append(
            f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | "
            f"{m['f1']:.3f} | {m['support']} |"
        )

    L += ["", "## Most frequent confusions", ""]
    for c in r["top_confusions"]:
        L.append(f"- {c['true']} misread as {c['pred']}: {c['count']}x")

    L += ["", "## Robustness slices", ""]
    for attr, buckets in r["slices"].items():
        L += [f"**{attr}**", "", "| value | n | accuracy |", "| --- | --- | --- |"]
        for k, v in buckets.items():
            L.append(f"| {k} | {v['n']} | {v['accuracy']:.3f} |")
        L.append("")

    L += [
        "## Risk-coverage",
        "",
        "How often the system is wrong, among the fraction of predictions it is "
        "most confident about. This is what justifies the accept threshold.",
        "",
        "| coverage | risk (error rate) | confidence threshold |",
        "| --- | --- | --- |",
    ]
    for pt in r["risk_coverage_summary"]:
        L.append(f"| {pt['coverage']:.2f} | {pt['risk']:.4f} | {pt['threshold']:.3f} |")

    if "baseline" in r:
        b = r["baseline"]
        L += [
            "",
            "## Baseline comparison",
            "",
            "| system | accuracy | params |",
            "| --- | --- | --- |",
            f"| DTW k-NN (no training) | {b['accuracy']:.3f} | 0 |",
            f"| {r['model']} | {r['accuracy_mean']:.3f} | {r['params']:,} |",
            "",
            f"Delta: **{r['accuracy_mean'] - b['accuracy']:+.3f}**. If this is small, "
            "the neural model is not earning its complexity and the report should say so.",
        ]

    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--checkpoint", default="artifacts/gru.pt")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--baseline", action="store_true", help="also run DTW k-NN")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    samples = load_dataset(args.data)
    frames = CONFIG.window.frames
    folds = leave_one_subject_out(samples)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build(ckpt["model"], FEATURE_DIM, CONFIG.train, NUM_CLASSES)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    T = float(ckpt.get("temperature", 1.0))

    all_logits, all_y, all_idx, fold_accs = [], [], [], []
    for _, te in folds:
        dl = DataLoader(SignDataset(samples, te, frames), batch_size=64)
        with torch.no_grad():
            lg = np.concatenate([model(x).numpy() for x, _ in dl])
        y = np.array([LABEL_TO_INDEX[samples[i].label] for i in te])
        all_logits.append(lg)
        all_y.append(y)
        all_idx += te
        fold_accs.append(float((softmax(lg / T).argmax(1) == y).mean()))

    logits = np.concatenate(all_logits)
    y_true = np.concatenate(all_y)
    probs_raw = softmax(logits)
    probs = softmax(logits / T)
    y_pred = probs.argmax(1)

    cm = confusion_matrix(y_true, y_pred)
    pc = per_class_metrics(cm, VOCAB)
    ece_raw, _ = expected_calibration_error(probs_raw, y_true)
    ece_cal, bins = expected_calibration_error(probs, y_true)
    curve = risk_coverage_curve(probs, y_true)

    result = {
        "protocol": "leave-one-subject-out",
        "model": ckpt["model"],
        "params": sum(p.numel() for p in model.parameters()),
        "n_folds": len(folds),
        "n_subjects": len({s.subject for s in samples}),
        "n_samples": len(samples),
        "accuracy_mean": float(np.mean(fold_accs)),
        "accuracy_std": float(np.std(fold_accs)),
        "accuracy_min": float(np.min(fold_accs)),
        "macro_f1": macro_f1(pc),
        "ece_raw": ece_raw,
        "ece_calibrated": ece_cal,
        "mce_calibrated": maximum_calibration_error(probs, y_true),
        "brier": brier_score(probs, y_true),
        "temperature": T,
        "per_class": pc,
        "confusion_matrix": cm.tolist(),
        "top_confusions": top_confusions(cm, VOCAB),
        "reliability_bins": bins,
        "risk_coverage_summary": [
            curve[int(len(curve) * f) - 1] for f in (0.25, 0.5, 0.75, 0.9, 1.0)
        ],
        "slices": {
            a: slice_metrics(samples, all_idx, y_true, y_pred, a)
            for a in ("lighting", "distance", "background", "dominant", "subject")
        },
        "latency": measure_latency(model, frames, FEATURE_DIM),
    }

    if args.baseline:
        print("running DTW k-NN baseline (slow, O(n^2) in the training set)...")
        accs = []
        for tr, te in folds:
            Xtr = np.stack([resample(samples[i].features, frames) for i in tr])
            ytr = np.array([LABEL_TO_INDEX[samples[i].label] for i in tr])
            Xte = np.stack([resample(samples[i].features, frames) for i in te])
            yte = np.array([LABEL_TO_INDEX[samples[i].label] for i in te])
            accs.append(float((DTWkNN().fit(Xtr, ytr).predict(Xte) == yte).mean()))
        result["baseline"] = {"name": "dtw_knn", "accuracy": float(np.mean(accs)),
                              "per_fold": accs}

    (out / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(out / "report.md", result)
    print(f"\nwrote {out/'evaluation.json'} and {out/'report.md'}")
    print(
        f"accuracy {result['accuracy_mean']:.3f} (worst subject {result['accuracy_min']:.3f}), "
        f"ECE {ece_raw:.3f} -> {ece_cal:.3f}"
    )


if __name__ == "__main__":
    main()
