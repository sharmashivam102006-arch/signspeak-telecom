"""Evaluation for the SignLSTM sign-recognition model.

Two entry points:
  * Library: evaluate_model(model, samples) / evaluate_weights(path, samples)
    — run a trained model over held-out samples and report per-class
    precision, recall, F1-score, support and a confusion matrix.
  * CLI: python -m app.ml.evaluation.evaluate --weights <path>
    [--samples-dir <dir>] [--out report.json]
    Loads recorded samples from SIGN_SAMPLES_DIR and evaluates the
    weights file against them. Prints the report as JSON.

Metrics are computed in pure Python (no sklearn dependency). A metric
is null when it is undefined for that class (e.g. precision when the
model never predicted the class, recall when the class has no held-out
samples) — null is reported honestly, never replaced with 0 or 1.
Samples are REAL recorded landmark sequences only; this module never
fabricates data. Test fixtures that exercise it are labelled as such.
"""
import argparse
import json
import os
import sys


def compute_metrics(y_true, y_pred, labels):
    """Per-class precision/recall/F1 + confusion matrix + accuracy.

    y_true / y_pred are lists of class indices into ``labels``.
    Returns {"per_class": {label: {"precision": float|None,
    "recall": float|None, "f1": float|None, "support": int}},
    "confusion_matrix": {"labels": [...], "matrix": [[int, ...], ...]},
    "accuracy": float|None, "n": int}.
    Confusion matrix rows are true labels, columns predicted labels.
    accuracy is None when n == 0.
    """
    n = len(y_true)
    k = len(labels)
    matrix = [[0] * k for _ in range(k)]
    for t, p in zip(y_true, y_pred):
        if 0 <= t < k and 0 <= p < k:
            matrix[t][p] += 1
    per_class = {}
    for i, label in enumerate(labels):
        tp = matrix[i][i]
        fp = sum(matrix[r][i] for r in range(k)) - tp
        fn = sum(matrix[i]) - tp
        support = sum(matrix[i])
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        if precision is not None and recall is not None and (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = None
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    correct = sum(matrix[i][i] for i in range(k))
    return {
        "per_class": per_class,
        "confusion_matrix": {"labels": list(labels), "matrix": matrix},
        "accuracy": (correct / n) if n else None,
        "n": n,
    }


def _normalise_samples(samples):
    """Accept (label, sequence) tuples or dicts with label/sequence keys."""
    out = []
    for s in samples:
        if isinstance(s, dict):
            out.append((s["label"], s["sequence"]))
        else:
            label, seq = s
            out.append((label, seq))
    return out


def evaluate_model(model, samples, labels=None):
    """Run a loaded torch model over held-out samples and report metrics.

    ``model`` is a SignLSTM in eval mode (or any callable returning
    logits for a (1, 30, 63) float tensor). ``samples`` is a list of
    (label, sequence) tuples or dicts. Raises ValueError when there are
    no samples to evaluate on.
    """
    import torch  # type: ignore

    from app.ml.models.sign_lstm import VOCABULARY
    from app.ml.preprocessing.normalize import sequence_to_features

    labels = list(labels or VOCABULARY)
    items = _normalise_samples(samples)
    if not items:
        raise ValueError("No held-out samples to evaluate on.")
    y_true, y_pred = [], []
    with torch.no_grad():
        for label, seq in items:
            feats = sequence_to_features(seq)
            x = torch.tensor([feats], dtype=torch.float32)
            logits = model(x)
            y_true.append(labels.index(label))
            y_pred.append(int(torch.softmax(logits, dim=-1)[0].argmax()))
    return compute_metrics(y_true, y_pred, labels)


def evaluate_weights(weights_path, samples, labels=None, hidden_dim=None):
    """Load weights from ``weights_path`` and evaluate on ``samples``.

    Raises FileNotFoundError when the weights file does not exist and
    ValueError when there are no samples. The model is built from the
    checkpoint's stored hidden_dim (bare state dicts default to 128).
    """
    import torch  # type: ignore

    from app.ml.models.sign_lstm import VOCABULARY, SignLSTM

    if not weights_path or not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"No trained weights at {weights_path!r}. Train first with "
            "POST /api/ai/sign/train (after recording real samples).")
    labels = list(labels or VOCABULARY)
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
        hidden_dim = int(checkpoint.get("hidden_dim", hidden_dim or 128))
    else:
        state = checkpoint
        hidden_dim = hidden_dim or 128
    model = SignLSTM(input_dim=63, num_classes=len(labels),
                     hidden_dim=hidden_dim)
    model.load_state_dict(state)
    model.eval()
    return evaluate_model(model, samples, labels=labels)


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Evaluate trained SignLSTM weights on recorded samples")
    p.add_argument("--weights", default=None,
                   help="Path to sign_lstm.pt (default: config SIGN_MODEL_PATH)")
    p.add_argument("--samples-dir", default=None,
                   help="Directory of recorded samples (default: config SIGN_SAMPLES_DIR)")
    p.add_argument("--out", default=None,
                   help="Optional path to write the JSON report to")
    args = p.parse_args(argv)

    from app import config
    from app.services import sign_data

    weights = args.weights or config.SIGN_MODEL_PATH
    if args.samples_dir:
        config.SIGN_SAMPLES_DIR = args.samples_dir
    samples = sign_data.load_all_detailed()
    if not samples:
        print("No recorded samples found — nothing to evaluate. "
              "Record samples via POST /api/ai/sign/collect first.")
        return 1
    try:
        report = evaluate_weights(weights, samples)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Evaluation failed: {exc}")
        return 1
    report["weights"] = weights
    report["samples_evaluated"] = len(samples)
    text = json.dumps(report, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"Wrote evaluation report to {args.out}")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
