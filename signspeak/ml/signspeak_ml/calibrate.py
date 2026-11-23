"""Confidence calibration, abstention, and threshold fitting.

The original spec puts a number on screen - "HELLO - 94%" - and gates the
speech output on it. That is exactly the right instinct and it is worth doing
properly, because a raw softmax score is not a probability. Modern networks are
systematically overconfident: a model whose top-1 softmax reads 0.94 is
typically right about 80% of the time, and the gap widens on subjects it has
never seen. Shipping an uncalibrated 94% to a Deaf user's conversation partner
is telling them something false about how much to trust the word they just
heard.

Three things live here, in the order you should apply them.

1. Temperature scaling. One scalar, fitted on held-out data, dividing the
   logits. It cannot change which class wins, so accuracy is untouched, and it
   reliably cuts expected calibration error by 3-5x. It is the highest
   value-per-line of code in this repo.

2. Open-set rejection. The classifier is closed-set: it must output one of the
   seven classes. A user waving, adjusting their glasses, or signing a word
   outside the vocabulary still gets a label, often a confident one. The
   trained NONE class handles what you anticipated; an energy-based score
   handles what you did not.

3. Threshold fitting from the risk-coverage curve. Pick the accept/confirm
   thresholds from an explicit error budget - "at most 2% of spoken outputs may
   be wrong" - instead of from taste. This turns the UX policy into something
   you can defend with a number.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------
class TemperatureScaler:
    """Fit a single T minimising NLL on a held-out split (Guo et al., 2017)."""

    def __init__(self) -> None:
        self.log_t = torch.zeros(1, requires_grad=True)

    @property
    def temperature(self) -> float:
        return float(self.log_t.detach().exp())

    def fit(self, logits: np.ndarray, labels: np.ndarray, max_iter: int = 200) -> "TemperatureScaler":
        lg = torch.as_tensor(logits, dtype=torch.float32)
        lb = torch.as_tensor(labels, dtype=torch.long)
        # Optimise log T so T stays positive without a constraint.
        opt = torch.optim.LBFGS([self.log_t], lr=0.05, max_iter=max_iter)

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(lg / self.log_t.exp(), lb)
            loss.backward()
            return loss

        opt.step(closure)
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return (np.asarray(logits, dtype=np.float32) / self.temperature).astype(np.float32)

    def probs(self, logits: np.ndarray) -> np.ndarray:
        return softmax(self.transform(logits))


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


# ---------------------------------------------------------------------------
# Calibration error
# ---------------------------------------------------------------------------
def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> tuple[float, list[dict]]:
    """ECE plus the per-bin table behind the reliability diagram.

    Report ECE before and after scaling. A typical uncalibrated small GRU on a
    signer-independent split sits at 0.10-0.20; after temperature scaling
    0.02-0.05 is a reasonable target. Quote both.
    """
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    table: list[dict] = []
    n = len(labels)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        cnt = int(m.sum())
        if cnt == 0:
            table.append({"lo": float(lo), "hi": float(hi), "n": 0, "acc": None, "conf": None})
            continue
        acc = float(correct[m].mean())
        avg = float(conf[m].mean())
        ece += (cnt / n) * abs(acc - avg)
        table.append({"lo": float(lo), "hi": float(hi), "n": cnt, "acc": acc, "conf": avg})
    return float(ece), table


def maximum_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    _, table = expected_calibration_error(probs, labels, n_bins)
    return max((abs(b["acc"] - b["conf"]) for b in table if b["n"] > 0), default=0.0)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), labels] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


# ---------------------------------------------------------------------------
# Open-set rejection
# ---------------------------------------------------------------------------
def energy_score(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Free energy -T * logsumexp(logits / T). Lower = more in-distribution.

    Cheaper and generally stronger than max-softmax for out-of-distribution
    detection (Liu et al., 2020), and it needs nothing at training time. Use it
    alongside the NONE class, not instead: NONE catches the movements you
    thought to record, energy catches the ones you did not.
    """
    lg = np.asarray(logits, dtype=np.float64) / temperature
    m = lg.max(axis=-1, keepdims=True)
    lse = (m + np.log(np.exp(lg - m).sum(axis=-1, keepdims=True))).squeeze(-1)
    return (-temperature * lse).astype(np.float32)


def fit_energy_threshold(
    in_dist_logits: np.ndarray, temperature: float, target_tpr: float = 0.95
) -> float:
    """Threshold admitting `target_tpr` of genuine in-vocabulary signs.

    Fitted on in-distribution data only, so it is honest about the case you
    actually have: you cannot enumerate everything a user might do in front of
    a camera, so do not pretend to have an OOD training set.
    """
    e = energy_score(in_dist_logits, temperature)
    return float(np.quantile(e, target_tpr))


# ---------------------------------------------------------------------------
# Risk-coverage and threshold fitting
# ---------------------------------------------------------------------------
def risk_coverage_curve(probs: np.ndarray, labels: np.ndarray) -> list[dict]:
    """Sort by confidence; report error rate among the top-k fraction kept.

    This is the curve that answers the question the product actually asks:
    "if I only speak the ones I'm most sure about, how often am I wrong, and
    how much do I refuse to say?" Accuracy alone cannot answer it.
    """
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(np.float64)
    order = np.argsort(-conf)
    c = correct[order]
    cum_err = np.cumsum(1.0 - c)
    n = len(labels)
    return [
        {
            "coverage": (k + 1) / n,
            "risk": float(cum_err[k] / (k + 1)),
            "threshold": float(conf[order][k]),
        }
        for k in range(n)
    ]


@dataclass
class Thresholds:
    accept: float
    confirm: float
    energy: float
    temperature: float
    achieved_risk: float
    achieved_coverage: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def fit_thresholds(
    probs: np.ndarray,
    labels: np.ndarray,
    logits: np.ndarray,
    temperature: float,
    max_risk: float = 0.02,
    confirm_risk: float = 0.15,
) -> Thresholds:
    """Choose accept/confirm thresholds for a stated error budget.

    `max_risk=0.02` means: of everything the system speaks aloud unprompted, at
    most 2% may be the wrong word. `confirm_risk=0.15` is the looser bar for
    "Possible: HELLO - repeat to confirm", where the user is explicitly told
    not to trust it yet.

    Both budgets are product decisions, not statistical ones. A banking call
    centre and a classroom want different numbers, and the honest thing is to
    expose them as settings rather than bake in 0.90.
    """
    curve = risk_coverage_curve(probs, labels)

    def threshold_for(budget: float) -> tuple[float, float, float]:
        best = None
        for pt in curve:
            if pt["risk"] <= budget:
                best = pt
        if best is None:
            # Even the single most confident prediction breaks the budget.
            top = curve[0]
            return 1.01, top["risk"], 0.0
        return best["threshold"], best["risk"], best["coverage"]

    accept, risk, coverage = threshold_for(max_risk)
    confirm, _, _ = threshold_for(confirm_risk)
    confirm = min(confirm, accept)

    return Thresholds(
        accept=float(accept),
        confirm=float(confirm),
        energy=fit_energy_threshold(logits, temperature),
        temperature=float(temperature),
        achieved_risk=float(risk),
        achieved_coverage=float(coverage),
    )


# ---------------------------------------------------------------------------
# Decision policy - the single place accept/confirm/reject is decided
# ---------------------------------------------------------------------------
@dataclass
class Decision:
    action: str          # "speak" | "confirm" | "reject"
    label: str
    confidence: float
    runner_up: str | None
    reason: str


def decide(
    probs: np.ndarray,
    logits: np.ndarray,
    vocab: list[str],
    th: Thresholds,
    none_index: int = 0,
) -> Decision:
    """Turn one prediction into one of three user-visible outcomes.

    Kept here, in Python, next to the thresholds that were fitted for it, and
    mirrored in `apps/web/lib/decision.ts`. Scattering this logic across UI
    components is how a system ends up speaking words it was supposed to
    withhold.
    """
    order = np.argsort(-probs)
    top, second = int(order[0]), int(order[1]) if len(order) > 1 else None
    conf = float(probs[top])
    runner_up = vocab[second] if second is not None else None

    if top == none_index:
        return Decision("reject", "NONE", conf, runner_up, "classified as background movement")

    e = float(energy_score(logits[None, :], th.temperature)[0])
    if e > th.energy:
        return Decision(
            "reject", vocab[top], conf, runner_up,
            "outside the vocabulary the model was trained on",
        )

    if conf >= th.accept:
        return Decision("speak", vocab[top], conf, runner_up, "above accept threshold")
    if conf >= th.confirm:
        return Decision("confirm", vocab[top], conf, runner_up, "ambiguous - ask the user to repeat")
    return Decision("reject", vocab[top], conf, runner_up, "below confirm threshold")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save(path: str | Path, th: Thresholds, metrics: dict) -> None:
    Path(path).write_text(
        json.dumps({"thresholds": th.to_dict(), "metrics": metrics}, indent=2), encoding="utf-8"
    )


def load(path: str | Path) -> tuple[Thresholds, dict]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return Thresholds(**d["thresholds"]), d.get("metrics", {})
