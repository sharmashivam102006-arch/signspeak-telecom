"""Standalone inference entry point.

    python -m app.ml.inference.predict --landmarks landmarks.json

landmarks.json: {"landmarks": [[x,y,z] x21]} or {"sequence": [[[x,y,z]] x frames]}.
Uses the trained PyTorch model when weights exist, otherwise falls back
to the demo rule-based classifier (clearly labelled in the output).
"""
import argparse
import json
import sys

from app.services.sign_model_service import predict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--landmarks", required=True, help="JSON file with landmarks/sequence")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    with open(args.landmarks) as f:
        data = json.load(f)
    result = predict(landmarks=data.get("landmarks"), sequence=data.get("sequence"), force=args.force)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
