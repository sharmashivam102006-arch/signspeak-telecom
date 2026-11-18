"""Training for SignLSTM.

Two entry points:
  * Library: train_from_samples(samples, out_path, epochs, ...) — used by
    POST /api/ai/sign/train. ``samples`` is a list of (label, sequence)
    tuples or dicts with label/sequence/signer_id keys, where each
    sequence is 5-30 frames of 21 [x, y, z] landmarks, as collected by
    POST /api/ai/sign/collect. Trains on real recorded landmarks only —
    this module never fabricates training data.
  * Validation split: by-signer when samples come from >= 2 signers
    (train on N signers, validate on a held-out signer — the honest
    test of whether the model generalises to a new person), falling
    back to a stratified random split with a single signer, and to no
    validation set when too few samples exist. The split used is always
    recorded in the result ("by_signer" | "random" | "none").
  * After training, the model is evaluated on the held-out set with
    app.ml.evaluation.evaluate (per-class precision/recall/F1 and a
    confusion matrix). When there is no validation data the result
    says val_available=False — metrics are never fabricated.
  * CLI: python -m app.ml.training.train --data data/signs.csv --epochs 30
    --out app/ml/models/sign_lstm.pt
    Dataset CSV format (one row per frame, frames grouped by sample):
        sample_id,label,frame,landmark_0_x,landmark_0_y,landmark_0_z,...,landmark_20_z
"""
import argparse
import os
import random
import sys


def _normalise_items(samples):
    """Normalise (label, sequence) tuples or dicts to dicts with
    label/sequence/signer_id keys. Missing signer_id becomes "anon"."""
    items = []
    for s in samples:
        if isinstance(s, dict):
            items.append({
                "label": s["label"],
                "sequence": s["sequence"],
                "signer_id": str(s.get("signer_id") or "anon"),
            })
        else:
            label, seq = s
            items.append({"label": label, "sequence": seq, "signer_id": "anon"})
    return items


def split_samples(samples, seed=42):
    """Split recorded samples into train/validation sets.

    Strategy, in order:
      1. "by_signer" — when samples come from >= 2 distinct signers,
         hold out one whole signer (the one covering the most labels,
         then the most samples, ties broken by signer id) and train on
         the rest. This is the honest generalisation test: validation
         measures recognition of a person the model never trained on.
         Only used when every validated label also has >= 1 training
         sample; otherwise fall through to the random split.
      2. "random" — stratified per label, ~20% held out (at least 1
         sample when the label has >= 3 samples).
      3. "none" — too few samples to hold anything out.

    Returns (train_items, val_items, split, train_signers, val_signers)
    where items are {"label", "sequence", "signer_id"} dicts.
    """
    items = _normalise_items(samples)
    rng = random.Random(seed)
    signers = sorted({it["signer_id"] for it in items})
    if len(signers) >= 2:
        def _held_out_key(signer):
            mine = [it for it in items if it["signer_id"] == signer]
            return (len({it["label"] for it in mine}), len(mine))
        # Most labels covered, then most samples; deterministic tie-break
        # on signer id (sorted order, take the last of the max group).
        best = max(signers, key=lambda s: (_held_out_key(s), s))
        val_items = [it for it in items if it["signer_id"] == best]
        train_items = [it for it in items if it["signer_id"] != best]
        val_labels = {it["label"] for it in val_items}
        train_labels = {it["label"] for it in train_items}
        if train_items and val_items and val_labels <= train_labels:
            train_signers = sorted({it["signer_id"] for it in train_items})
            return train_items, val_items, "by_signer", train_signers, [best]
    # Stratified random fallback (also the single-signer path).
    by_label = {}
    for it in items:
        by_label.setdefault(it["label"], []).append(it)
    train_items, val_items = [], []
    for label, seqs in by_label.items():
        seqs = list(seqs)
        rng.shuffle(seqs)
        n_val = max(1, round(len(seqs) * 0.2)) if len(seqs) >= 3 else 0
        val_items.extend(seqs[:n_val])
        train_items.extend(seqs[n_val:])
    if val_items:
        train_signers = sorted({it["signer_id"] for it in train_items})
        val_signers = sorted({it["signer_id"] for it in val_items})
        return train_items, val_items, "random", train_signers, val_signers
    return (train_items, val_items, "none",
            sorted({it["signer_id"] for it in train_items}), [])


def train_from_samples(samples, out_path, epochs=30, hidden_dim=128,
                       progress_cb=None, seed=42):
    """Train SignLSTM on recorded samples (tuples or dicts, see above).

    Returns {"val_accuracy": float|None, "train_samples": int,
    "val_samples": int, "epochs": int, "out_path": str,
    "split": "by_signer"|"random"|"none", "val_available": bool,
    "per_class": {label: {precision, recall, f1, support}}|None,
    "confusion_matrix": {labels, matrix}|None,
    "train_signers": [str], "val_signers": [str]}.
    val_accuracy/per_class/confusion_matrix are None and
    val_available is False when no validation data exists — the caller
    must say so honestly instead of inventing a number.
    """
    import torch  # type: ignore
    from torch.utils.data import DataLoader, TensorDataset

    from app.ml.evaluation.evaluate import evaluate_model
    from app.ml.models.sign_lstm import VOCABULARY, SignLSTM
    from app.ml.preprocessing.normalize import sequence_to_features

    if not samples:
        raise ValueError("No training samples. Record samples first via POST /api/ai/sign/collect.")
    items = _normalise_items(samples)
    labels = {it["label"] for it in items}
    if len(labels) < 2:
        raise ValueError(
            f"Need samples for at least 2 different signs to train, got {sorted(labels)}. "
            "Record more signs via POST /api/ai/sign/collect.")

    train_items, val_items, split, train_signers, val_signers = split_samples(items, seed=seed)
    if not train_items:
        raise ValueError("Not enough samples to train (every sample was held out for validation).")

    def _tensors(items):
        X = [sequence_to_features(it["sequence"]) for it in items]
        y = [VOCABULARY.index(it["label"]) for it in items]
        return (torch.tensor(X, dtype=torch.float32),
                torch.tensor(y, dtype=torch.long))

    X_train, y_train = _tensors(train_items)
    X_val, y_val = _tensors(val_items) if val_items else (None, None)

    model = SignLSTM(num_classes=len(VOCABULARY), hidden_dim=hidden_dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=8, shuffle=True)

    val_accuracy = None
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item()
        if X_val is not None:
            model.eval()
            with torch.no_grad():
                preds = model(X_val).argmax(dim=1)
                val_accuracy = float((preds == y_val).float().mean())
        if progress_cb:
            progress_cb(epoch + 1, epochs, total / len(loader), val_accuracy)

    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    tmp_path = out_path + ".tmp"
    # Save a checkpoint dict, not a bare state dict: inference must build
    # the same architecture (hidden_dim) the weights were trained with.
    torch.save({
        "state_dict": model.state_dict(),
        "hidden_dim": hidden_dim,
        "input_dim": 63,
        "num_classes": len(VOCABULARY),
    }, tmp_path)
    os.replace(tmp_path, out_path)
    # Evaluate the trained model on the held-out set (when one exists)
    # and store the full report — per-class precision/recall/F1 and the
    # confusion matrix — so the job result can show real validation
    # numbers instead of a single accuracy. No val set => val_available
    # False and every metric None; never fabricate them.
    per_class = None
    confusion_matrix = None
    if val_items:
        model.eval()
        report = evaluate_model(model, val_items)
        per_class = report["per_class"]
        confusion_matrix = report["confusion_matrix"]
        val_accuracy = report["accuracy"]
    return {
        "val_accuracy": val_accuracy,
        "train_samples": len(train_items),
        "val_samples": len(val_items),
        "epochs": epochs,
        "out_path": out_path,
        "split": split,
        "val_available": bool(val_items),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix,
        "train_signers": train_signers,
        "val_signers": val_signers,
    }


def main():
    p = argparse.ArgumentParser(description="Train SignLSTM on landmark CSV data")
    p.add_argument("--data", default=None, help="Path to dataset CSV (see module docstring)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--out", default="app/ml/models/sign_lstm.pt")
    p.add_argument("--hidden", type=int, default=128)
    args = p.parse_args()

    if not args.data or not os.path.exists(args.data):
        print(__doc__)
        print("No dataset provided/found — nothing was trained.")
        print("To train: collect 30-frame landmark sequences per sign, save as CSV")
        print("in the format above, then re-run with --data <path>.")
        return 0
    try:
        import csv
        from collections import defaultdict
    except ImportError as e:
        print(f"Missing dependency for training: {e}.")
        return 1

    samples = defaultdict(list)  # (sample_id, label) -> list of frames
    with open(args.data, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["sample_id"], row["label"])
            frame = [[float(row[f"landmark_{i}_{a}"]) for a in "xyz"] for i in range(21)]
            samples[key].append((int(row["frame"]), frame))
    data = []
    for (_sid, label), frames in samples.items():
        frames.sort()
        data.append((label, [fr for _, fr in frames]))
    if not data:
        print("Dataset contained no usable samples.")
        return 1
    try:
        result = train_from_samples(data, args.out, epochs=args.epochs,
                                    hidden_dim=args.hidden)
    except Exception as exc:
        print(f"Training failed: {exc}")
        return 1
    print(f"Saved trained weights to {args.out} "
          f"(split={result['split']}, val_accuracy={result['val_accuracy']}, "
          f"train={result['train_samples']}, val={result['val_samples']})")
    if result["per_class"]:
        for label, m in result["per_class"].items():
            p = f"{m['precision']:.2f}" if m["precision"] is not None else "n/a"
            r = f"{m['recall']:.2f}" if m["recall"] is not None else "n/a"
            f = f"{m['f1']:.2f}" if m["f1"] is not None else "n/a"
            print(f"  {label:10s} precision={p} recall={r} f1={f} support={m['support']}")
    else:
        print("No validation data — no per-class metrics (val_available=False).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
