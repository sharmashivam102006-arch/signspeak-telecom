"""Landmark normalisation (pure Python, no MediaPipe dependency).

MediaPipe Hands returns 21 landmarks per hand, each [x, y, z] in
image coordinates. Normalisation makes sequences translation- and
scale-invariant so the temporal model sees hand shape, not position:
  1. Translate so the wrist (landmark 0) is at the origin.
  2. Scale by the distance wrist -> middle-finger MCP (landmark 9),
     falling back to the bounding-box size.
See app/ml/preprocessing/normalize.py for the shared implementation.
"""
import math
from typing import List

from app.ml.preprocessing.normalize import normalize_landmarks, flatten_landmarks

__all__ = ["normalize_landmarks", "flatten_landmarks", "landmarks_valid"]


def landmarks_valid(landmarks) -> bool:
    """21 landmarks, each [x, y, z] numeric."""
    if not isinstance(landmarks, (list, tuple)) or len(landmarks) != 21:
        return False
    for lm in landmarks:
        if not isinstance(lm, (list, tuple)) or len(lm) < 2:
            return False
        try:
            float(lm[0]); float(lm[1])
        except (TypeError, ValueError):
            return False
    return True
