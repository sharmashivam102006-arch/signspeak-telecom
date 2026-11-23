"""Temporal classifiers, plus the baseline you have to beat.

Read this before picking a model: with six well-separated signs in a
body-anchored feature space, the k-NN/DTW baseline below will land somewhere
around 90-96% signer-dependent accuracy. If your GRU does not clearly beat it
on the *signer-independent* split, the GRU is decoration and the honest thing
to do is say so in the report. Reviewers notice when a deep model is chosen for
the sound of it.

The comparison itself is the contribution. Three architectures on identical
features, identical splits, identical calibration, with parameter counts and
inference latency alongside accuracy, is a real result at this scale. A single
unjustified Transformer is not.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .config import NUM_CLASSES, TrainConfig


# ---------------------------------------------------------------------------
# Baseline: DTW + k-NN. No training, no GPU, no gradients.
# ---------------------------------------------------------------------------
class DTWkNN:
    """Dynamic time warping nearest-neighbour over feature sequences.

    Also used as-is for the few-shot path in the browser, where a user records
    ~10 examples of a new sign and expects it to work immediately. Any system
    that can only recognise what it was trained on months ago is not an
    accessibility tool, it is a demo, so this path is not throwaway.

    Cost is O(n_train * T^2 * D). At T=32 and a few hundred templates that is
    single-digit milliseconds - fine live. It stops being fine around a few
    thousand templates, which is roughly where you should switch to the neural
    model anyway.
    """

    def __init__(self, k: int = 3, band: int = 6) -> None:
        self.k = k
        self.band = band  # Sakoe-Chiba band: caps warping, ~4x faster, more accurate
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DTWkNN":
        self.X, self.y = X.astype(np.float32), y.astype(np.int64)
        return self

    def _dtw(self, a: np.ndarray, b: np.ndarray) -> float:
        n, m = a.shape[0], b.shape[0]
        # squared euclidean cost matrix
        cost = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        inf = np.inf
        D = np.full((n + 1, m + 1), inf, dtype=np.float32)
        D[0, 0] = 0.0
        for i in range(1, n + 1):
            lo = max(1, i - self.band)
            hi = min(m, i + self.band)
            for j in range(lo, hi + 1):
                D[i, j] = cost[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
        return float(D[n, m] / (n + m))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.X is not None and self.y is not None, "fit() first"
        out = np.zeros((X.shape[0], NUM_CLASSES), dtype=np.float32)
        for i, q in enumerate(X):
            d = np.array([self._dtw(q, t) for t in self.X], dtype=np.float32)
            idx = np.argsort(d)[: self.k]
            # Inverse-distance vote. Not a calibrated probability, and treated
            # as such everywhere downstream.
            w = 1.0 / (d[idx] + 1e-6)
            for label, weight in zip(self.y[idx], w):
                out[i, label] += weight
            out[i] /= out[i].sum() + 1e-9
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(1)


# ---------------------------------------------------------------------------
# Neural models
# ---------------------------------------------------------------------------
class MaskedMeanMax(nn.Module):
    """Concatenated mean+max pooling over time. Cheap, and consistently a
    point or two better than taking the last hidden state, which throws away
    the middle of the sign."""

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=-1)


class GRUClassifier(nn.Module):
    """Bidirectional GRU. The sensible default at this data scale.

    Bidirectional is legitimate here *because we classify completed segments*,
    not a live prefix. The moment you move to streaming early-commit
    classification this must become unidirectional, and the accuracy you quote
    from the bidirectional model stops being the accuracy users experience.
    """

    def __init__(self, in_dim: int, cfg: TrainConfig, n_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.rnn = nn.GRU(
            in_dim,
            cfg.hidden,
            num_layers=cfg.layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.dropout if cfg.layers > 1 else 0.0,
        )
        self.pool = MaskedMeanMax()
        self.head = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden * 4, cfg.hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(self.norm(x))
        return self.head(self.pool(h))


class TCNClassifier(nn.Module):
    """Dilated temporal CNN. Parallel over time, so the fastest of the three at
    inference, and the easiest to convert cleanly to ONNX/WebNN."""

    def __init__(self, in_dim: int, cfg: TrainConfig, n_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        c = cfg.hidden
        self.norm = nn.LayerNorm(in_dim)
        layers: list[nn.Module] = []
        prev = in_dim
        for d in (1, 2, 4, 8):
            layers += [
                nn.Conv1d(prev, c, kernel_size=3, padding=d, dilation=d),
                nn.BatchNorm1d(c),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            ]
            prev = c
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(c * 2, c), nn.GELU(), nn.Linear(c, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.tcn(self.norm(x).transpose(1, 2))
        return self.head(torch.cat([h.mean(-1), h.max(-1).values], dim=-1))


class TransformerClassifier(nn.Module):
    """Small encoder with learned positional embeddings.

    Include it for the comparison, but expect it to lose. Self-attention wants
    more data than a six-sign dataset collected by two students will contain,
    and "we tried it and it underperformed at this data scale" is a finding,
    not a failure.
    """

    def __init__(
        self, in_dim: int, cfg: TrainConfig, n_classes: int = NUM_CLASSES, max_len: int = 64
    ) -> None:
        super().__init__()
        d = cfg.hidden
        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, d))
        self.pos = nn.Parameter(torch.randn(1, max_len, d) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=4,
            dim_feedforward=d * 2,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        # enable_nested_tensor is incompatible with norm_first and only warns;
        # disabling it keeps the logs readable and changes nothing numerically.
        self.enc = nn.TransformerEncoder(layer, num_layers=cfg.layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d * 2), nn.Linear(d * 2, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        h = h + self.pos[:, : h.shape[1]]
        h = self.enc(h)
        return self.head(torch.cat([h.mean(1), h.max(1).values], dim=-1))


MODELS = {
    "gru": GRUClassifier,
    "tcn": TCNClassifier,
    "transformer": TransformerClassifier,
}


def build(name: str, in_dim: int, cfg: TrainConfig, n_classes: int = NUM_CLASSES) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    return MODELS[name](in_dim, cfg, n_classes)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
