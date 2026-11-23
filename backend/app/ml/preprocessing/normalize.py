"""Landmark preprocessing: normalise 21 MediaPipe hand landmarks.

Pure Python (no numpy/torch required) so the API can run without ML deps.
Steps: translate wrist (0) to origin, scale by wrist->middle-MCP (9)
distance (fallback: bounding-box diagonal), then flatten to 63 floats.
"""
import math
from typing import List


def normalize_landmarks(landmarks: List[List[float]]) -> List[List[float]]:
    lm = [[float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 else 0.0] for p in landmarks]
    wx, wy, wz = lm[0]
    translated = [[x - wx, y - wy, z - wz] for x, y, z in lm]
    scale = math.dist(translated[0], translated[9])  # wrist -> middle MCP
    if scale < 1e-6:
        xs = [p[0] for p in translated]; ys = [p[1] for p in translated]
        scale = math.hypot(max(xs) - min(xs), max(ys) - min(ys)) or 1.0
    return [[x / scale, y / scale, z / scale] for x, y, z in translated]


def flatten_landmarks(landmarks: List[List[float]]) -> List[float]:
    return [v for p in normalize_landmarks(landmarks) for v in p]


def sequence_to_features(sequence: List[List[List[float]]], seq_len: int = 30) -> List[List[float]]:
    """Normalise each frame and pad/truncate to seq_len frames (63 floats each)."""
    feats = [flatten_landmarks(f) for f in sequence]
    if len(feats) >= seq_len:
        return feats[-seq_len:]
    pad = [[0.0] * 63] * (seq_len - len(feats))
    return pad + feats
