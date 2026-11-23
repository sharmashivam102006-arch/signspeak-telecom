"""Training entry point.

Default protocol is leave-one-subject-out, because that is the only number
worth putting in the report. `--protocol signer-dependent` exists so you can
quote the inflated figure *next to* the honest one and show the reader the gap.

    python -m signspeak_ml.train --data data/raw --model gru
    python -m signspeak_ml.train --data data/raw --model gru --protocol signer-dependent
    python -m signspeak_ml.train --data data/raw --compare      # all three models
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .calibrate import (
    TemperatureScaler,
    expected_calibration_error,
    fit_thresholds,
    softmax,
)
from .config import CONFIG, NONE_INDEX, NUM_CLASSES, VOCAB, TrainConfig
from .dataset import (
    Augment,
    SignDataset,
    class_weights,
    describe,
    leave_one_subject_out,
    load_dataset,
    signer_dependent_split,
    subject_stratified_split,
)
from .features import FEATURE_DIM
from .models import build, count_params


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def collect_logits(model: nn.Module, loader: DataLoader, dev: torch.device):
    model.eval()
    L, Y = [], []
    for x, y in loader:
        L.append(model(x.to(dev)).cpu().numpy())
        Y.append(y.numpy())
    return np.concatenate(L), np.concatenate(Y)


def train_one(
    samples,
    train_idx: list[int],
    test_idx: list[int],
    model_name: str,
    cfg: TrainConfig,
    dev: torch.device,
    verbose: bool = True,
) -> dict:
    """Train on `train_idx`, calibrate on a subject-disjoint slice of it,
    evaluate on `test_idx`."""
    set_seed(cfg.seed)

    # Calibration split is carved out of TRAIN, by subject. Fitting temperature
    # on the test set would be leakage; fitting it on data the model trained on
    # would be useless.
    sub_train_rel, sub_cal_rel = subject_stratified_split(
        [samples[i] for i in train_idx], val_subjects=1, seed=cfg.seed
    )
    sub_train = [train_idx[i] for i in sub_train_rel]
    sub_cal = [train_idx[i] for i in sub_cal_rel]
    if not sub_cal:                       # too few subjects to spare one
        sub_train, sub_cal = train_idx, train_idx

    frames = CONFIG.window.frames
    aug = Augment(cfg)
    dl_train = DataLoader(
        SignDataset(samples, sub_train, frames, augment=aug),
        batch_size=cfg.batch_size, shuffle=True, drop_last=False,
    )
    dl_cal = DataLoader(SignDataset(samples, sub_cal, frames), batch_size=64)
    dl_test = DataLoader(SignDataset(samples, test_idx, frames), batch_size=64)

    model = build(model_name, FEATURE_DIM, cfg, NUM_CLASSES).to(dev)
    w = torch.tensor(class_weights(samples, sub_train, NUM_CLASSES), device=dev)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=cfg.label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    best_loss, best_state, bad = float("inf"), None, 0
    for epoch in range(cfg.epochs):
        model.train()
        total = 0.0
        for x, y in dl_train:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach()) * x.size(0)
        sched.step()

        cal_logits, cal_y = collect_logits(model, dl_cal, dev)
        val_loss = float(
            nn.functional.cross_entropy(
                torch.tensor(cal_logits), torch.tensor(cal_y)
            )
        )
        if val_loss < best_loss - 1e-4:
            best_loss, bad = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                if verbose:
                    print(f"    early stop @ epoch {epoch}")
                break
        if verbose and epoch % 10 == 0:
            print(f"    epoch {epoch:3d}  train {total/max(len(sub_train),1):.4f}  cal {val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # --- calibration -------------------------------------------------------
    cal_logits, cal_y = collect_logits(model, dl_cal, dev)
    scaler = TemperatureScaler().fit(cal_logits, cal_y)

    # --- test --------------------------------------------------------------
    test_logits, test_y = collect_logits(model, dl_test, dev)
    raw_probs = softmax(test_logits)
    cal_probs = scaler.probs(test_logits)

    ece_raw, _ = expected_calibration_error(raw_probs, test_y)
    ece_cal, bins = expected_calibration_error(cal_probs, test_y)

    th = fit_thresholds(scaler.probs(cal_logits), cal_y, cal_logits, scaler.temperature)

    acc = float((cal_probs.argmax(1) == test_y).mean())
    # Accuracy on real signs only - NONE is easy and dilutes the number upward.
    mask = test_y != NONE_INDEX
    acc_signs = float((cal_probs.argmax(1)[mask] == test_y[mask]).mean()) if mask.any() else 0.0

    return {
        "model": model_name,
        "params": count_params(model),
        "accuracy": acc,
        "accuracy_signs_only": acc_signs,
        "ece_raw": ece_raw,
        "ece_calibrated": ece_cal,
        "temperature": scaler.temperature,
        "thresholds": th.to_dict(),
        "reliability_bins": bins,
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "test_logits": test_logits,
        "test_labels": test_y,
        "n_train": len(sub_train),
        "n_test": len(test_idx),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a SignSpeak sign classifier")
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--model", default="gru", choices=["gru", "tcn", "transformer"])
    ap.add_argument("--compare", action="store_true", help="train all three and tabulate")
    ap.add_argument(
        "--protocol",
        default="loso",
        choices=["loso", "signer-dependent"],
        help="loso = leave-one-subject-out (report this one)",
    )
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = device()

    samples = load_dataset(args.data)
    audit = describe(samples)
    print(json.dumps(audit, indent=2))
    (out / "dataset_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    if audit["subjects"] < 3 and args.protocol == "loso":
        raise SystemExit(
            f"\nSTOP. {audit['subjects']} subject(s) in the dataset.\n"
            "Leave-one-subject-out is meaningless here and a signer-dependent number\n"
            "will be inflated by 15-30 points. Collect from 6-8 people before you\n"
            "quote any accuracy figure. Re-run with --protocol signer-dependent only\n"
            "if you are explicitly labelling the result as signer-dependent.\n"
        )

    cfg = CONFIG.train
    if args.epochs:
        cfg = TrainConfig(**{**cfg.__dict__, "epochs": args.epochs})

    model_names = ["gru", "tcn", "transformer"] if args.compare else [args.model]
    folds = (
        leave_one_subject_out(samples)
        if args.protocol == "loso"
        else [signer_dependent_split(samples)]
    )

    summary: dict[str, dict] = {}
    for name in model_names:
        fold_results = []
        t0 = time.time()
        for fi, (tr, te) in enumerate(folds):
            held = sorted({samples[i].subject for i in te})
            print(f"\n[{name}] fold {fi + 1}/{len(folds)}  held-out: {held}")
            r = train_one(samples, tr, te, name, cfg, dev, verbose=True)
            print(
                f"    acc {r['accuracy']:.3f} | signs-only {r['accuracy_signs_only']:.3f} "
                f"| ECE {r['ece_raw']:.3f} -> {r['ece_calibrated']:.3f} (T={r['temperature']:.2f})"
            )
            fold_results.append(r)

        accs = np.array([r["accuracy"] for r in fold_results])
        eces = np.array([r["ece_calibrated"] for r in fold_results])
        summary[name] = {
            "params": fold_results[0]["params"],
            "accuracy_mean": float(accs.mean()),
            "accuracy_std": float(accs.std()),
            "accuracy_min": float(accs.min()),   # the worst signer is the real story
            "ece_mean": float(eces.mean()),
            "train_seconds": time.time() - t0,
            "folds": [
                {k: v for k, v in r.items()
                 if k not in ("state_dict", "test_logits", "test_labels", "reliability_bins")}
                for r in fold_results
            ],
        }

        # Keep the best fold's weights + calibration for export.
        best = max(fold_results, key=lambda r: r["accuracy"])
        torch.save(
            {
                "state_dict": best["state_dict"],
                "model": name,
                "feature_dim": FEATURE_DIM,
                "frames": CONFIG.window.frames,
                "vocab": VOCAB,
                "thresholds": best["thresholds"],
                "temperature": best["temperature"],
            },
            out / f"{name}.pt",
        )
        np.savez(
            out / f"{name}_test_logits.npz",
            logits=best["test_logits"],
            labels=best["test_labels"],
        )

    (out / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"{'model':<14}{'params':>10}{'acc mean':>11}{'acc std':>10}{'worst':>9}{'ECE':>8}")
    print("-" * 78)
    for name, s in summary.items():
        print(
            f"{name:<14}{s['params']:>10,}{s['accuracy_mean']:>11.3f}"
            f"{s['accuracy_std']:>10.3f}{s['accuracy_min']:>9.3f}{s['ece_mean']:>8.3f}"
        )
    print("=" * 78)
    print(
        "\nIf the spread (std) across subjects is larger than the gap between models,\n"
        "the model choice is noise and the dataset is the bottleneck. Say that in the\n"
        "report rather than declaring a winner."
    )


if __name__ == "__main__":
    main()
