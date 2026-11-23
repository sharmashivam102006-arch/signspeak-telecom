"""Export a trained checkpoint to ONNX plus a runtime config for the browser.

Everything the client needs to behave identically to the trainer - vocabulary,
frame count, feature dimension, temperature, thresholds, segmenter constants -
is written into one JSON next to the model. The browser reads that file rather
than hardcoding anything, because a threshold that drifts between the paper and
the product makes both wrong.

    python -m signspeak_ml.export_onnx --checkpoint artifacts/gru.pt \\
        --out ../apps/web/public/models
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .config import CONFIG, SPEECH_TEMPLATES, VOCAB
from .features import FEATURE_DIM, STATIC_DIM
from .models import build


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="artifacts/gru.pt")
    ap.add_argument("--out", default="../apps/web/public/models")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build(ckpt["model"], FEATURE_DIM, CONFIG.train, len(VOCAB))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    frames = int(ckpt.get("frames", CONFIG.window.frames))
    dummy = torch.randn(1, frames, FEATURE_DIM)
    onnx_path = out / "sign_classifier.onnx"

    export_kwargs = dict(
        input_names=["features"],
        output_names=["logits"],
        # Batch is dynamic; time is fixed, because the segmenter always hands
        # us exactly `frames` after resampling. A dynamic time axis here would
        # silently permit the client to send unresampled input.
        dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
    )

    # torch >= 2.6 routes through the dynamo exporter by default, which needs
    # `onnxscript`. RNNs still export more reliably through the legacy
    # TorchScript path, so try that first and fall back rather than making the
    # build depend on which torch happens to be installed.
    try:
        torch.onnx.export(model, dummy, str(onnx_path), dynamo=False,
                          do_constant_folding=True, **export_kwargs)
    except TypeError:
        # Older torch has no `dynamo` argument at all.
        torch.onnx.export(model, dummy, str(onnx_path),
                          do_constant_folding=True, **export_kwargs)
    except Exception as exc:  # dynamo path unavailable or failed
        print(f"legacy ONNX export failed ({exc}); retrying with the dynamo exporter")
        torch.onnx.export(model, dummy, str(onnx_path), **export_kwargs)

    runtime = {
        "version": 1,
        "model": ckpt["model"],
        "onnx": onnx_path.name,
        "vocab": VOCAB,
        "speechTemplates": SPEECH_TEMPLATES,
        "frames": frames,
        "featureDim": FEATURE_DIM,
        "staticDim": STATIC_DIM,
        "temperature": float(ckpt.get("temperature", 1.0)),
        "thresholds": ckpt.get("thresholds", CONFIG.decision.__dict__),
        "segmenter": CONFIG.segmenter.__dict__,
        "window": CONFIG.window.__dict__,
        "decision": CONFIG.decision.__dict__,
    }
    (out / "runtime_config.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")

    # Parity check: PyTorch vs onnxruntime on the same input. If this drifts,
    # the browser is running a different model from the one you evaluated.
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        got = sess.run(None, {"features": dummy.numpy()})[0]
        with torch.no_grad():
            want = model(dummy).numpy()
        delta = float(np.abs(got - want).max())
        print(f"onnx parity: max |delta| = {delta:.2e}")
        if delta > 1e-3:
            raise SystemExit("ONNX output diverges from PyTorch - do not ship this.")
    except ImportError:
        print("onnxruntime not installed; skipped parity check (install it before shipping)")

    print(f"wrote {onnx_path} ({onnx_path.stat().st_size/1024:.0f} KB)")
    print(f"wrote {out/'runtime_config.json'}")


if __name__ == "__main__":
    main()
