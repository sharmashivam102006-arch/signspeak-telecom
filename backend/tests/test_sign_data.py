"""Tests for multi-signer sign collection, by-signer splits and evaluation.

Run: python -m pytest tests/test_sign_data.py  (from backend/)

All landmark sequences here are SYNTHETIC FIXTURES (jittered demo poses
built by _pose/_pose_seq below) — they exercise the plumbing only.
They are NOT training data, NOT real signs, and must never be shipped
as model weights. Real recognition requires real human samples; see
backend/docs/DATASET.md.
"""
import json
import os
import sys

os.environ.setdefault("DATABASE_URL", "sqlite:///./test_signspeak.db")
os.environ.setdefault("SECRET_KEY", "test-secret")
os.environ.setdefault("LIVEKIT_URL", "ws://localhost:7880")
os.environ.setdefault("LIVEKIT_API_KEY", "devkey")
os.environ.setdefault("LIVEKIT_API_SECRET", "secret")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient

from app.main import app  # noqa: E402

client = TestClient(app)

LABELS = ["Hello", "Yes", "No", "Help", "Thank You", "Emergency"]


def _auth_headers():
    import uuid as _uuid
    email = f"signdata-{_uuid.uuid4().hex[:8]}@example.com"
    r = client.post("/api/auth/register", json={
        "email": email, "name": "Sign Data Tester", "password": "password123"})
    assert r.status_code == 201, r.text
    body = r.json()
    return {"Authorization": f"Bearer {body['access_token']}"}, body["user"]["id"]


# ---------------------------------------------------------------- fixtures
def _pose(kind: str):
    """One 21-landmark frame for a demo pose — SYNTHETIC FIXTURE ONLY."""
    lm = [[0.5, 0.9, 0.0]] * 21
    lm = [list(p) for p in lm]
    lm[9] = [0.5, 0.7, 0.0]  # middle MCP (scale anchor)
    up = {"Hello": [8, 12, 16, 20], "Yes": [8], "No": [],
          "Help": [8, 12], "Thank You": [], "Emergency": [20]}[kind]
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        lm[pip] = [0.5, 0.6, 0.0]
        lm[tip] = [0.5, 0.4, 0.0] if tip in up else [0.5, 0.8, 0.0]
    lm[3] = [0.5, 0.6, 0.0]
    lm[4] = [0.5, 0.4, 0.0] if kind == "Thank You" else [0.5, 0.7, 0.0]
    return lm


def _pose_seq(kind: str, n: int = 10, seed: int = 0):
    """n jittered frames of a pose — SYNTHETIC FIXTURE ONLY."""
    import random
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        frame = _pose(kind)
        out.append([[x + rng.uniform(-0.01, 0.01),
                     y + rng.uniform(-0.01, 0.01), z]
                    for x, y, z in frame])
    return out


def _fixture_samples(signers, per_label=5):
    """SYNTHETIC FIXTURE samples: {label, sequence, signer_id} dicts."""
    out = []
    for si, signer in enumerate(signers):
        for label in LABELS:
            for i in range(per_label):
                out.append({"label": label,
                            "sequence": _pose_seq(label, n=10, seed=1000 * si + i),
                            "signer_id": signer})
    return out


# ---------------------------------------------------------------- collect auth
def test_collect_requires_auth(monkeypatch, tmp_path):
    from app import config
    monkeypatch.setattr(config, "SIGN_SAMPLES_DIR", str(tmp_path / "samples"))
    seq = _pose_seq("Hello", n=10, seed=1)
    r = client.post("/api/ai/sign/collect", json={"label": "Hello", "sequence": seq})
    assert r.status_code == 401, r.text
    r = client.post("/api/ai/sign/collect", json={"label": "Hello", "sequence": seq},
                    headers={"Authorization": "Bearer not-a-real-token"})
    assert r.status_code == 401, r.text


def test_collect_signer_id_from_user_not_client(monkeypatch, tmp_path):
    """signer_id comes from the JWT. A client-supplied signer_id is ignored."""
    from app import config
    from app.services import sign_data
    monkeypatch.setattr(config, "SIGN_SAMPLES_DIR", str(tmp_path / "samples"))
    headers, user_id = _auth_headers()
    seq = _pose_seq("Hello", n=10, seed=2)
    # Attacker tries to attribute the sample to someone else.
    r = client.post("/api/ai/sign/collect",
                    json={"label": "Hello", "sequence": seq,
                          "signer_id": "someone-else", "language": "ISL"},
                    headers=headers)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["signer_id"] == user_id
    assert body["signer_id"] != "someone-else"
    assert body["language"] == "isl"  # language IS client-settable, lower-cased
    stored = sign_data.load_all_detailed()
    assert len(stored) == 1
    assert stored[0]["signer_id"] == user_id
    assert stored[0]["language"] == "isl"
    counts = sign_data.counts()
    assert counts["samples_per_signer"] == {user_id: 1}
    assert counts["total_signers"] == 1
    assert counts["languages"] == {"isl": 1}


def test_collect_rejects_unknown_label_still(monkeypatch, tmp_path):
    from app import config
    monkeypatch.setattr(config, "SIGN_SAMPLES_DIR", str(tmp_path / "samples"))
    headers, _ = _auth_headers()
    r = client.post("/api/ai/sign/collect",
                    json={"label": "Wave", "sequence": _pose_seq("Hello", n=10)},
                    headers=headers)
    assert r.status_code == 400, r.text


# ---------------------------------------------------------------- counts
def test_counts_per_signer_and_language(monkeypatch, tmp_path):
    from app import config
    from app.services import sign_data
    monkeypatch.setattr(config, "SIGN_SAMPLES_DIR", str(tmp_path / "samples"))
    sign_data.save_sample("Hello", _pose_seq("Hello", seed=1), signer_id="alice")
    sign_data.save_sample("Hello", _pose_seq("Hello", seed=2), signer_id="alice")
    sign_data.save_sample("Hello", _pose_seq("Hello", seed=3), signer_id="bob",
                          language="ISL")
    sign_data.save_sample("Yes", _pose_seq("Yes", seed=4), signer_id="bob")
    c = sign_data.counts()
    assert c["samples_per_sign"] == {"Hello": 3, "Yes": 1, "No": 0,
                                     "Help": 0, "Thank You": 0, "Emergency": 0}
    assert c["samples_per_signer"] == {"alice": 2, "bob": 2}
    assert c["samples_per_sign_by_signer"]["Hello"] == {"alice": 2, "bob": 1}
    assert c["samples_per_sign_by_signer"]["Yes"] == {"bob": 1}
    assert c["languages"] == {"generic": 3, "isl": 1}
    assert c["signers"] == ["alice", "bob"]
    assert c["total_signers"] == 2
    assert c["total"] == 4


def test_legacy_sample_defaults(monkeypatch, tmp_path):
    """Sample files written before signer/language tracking default to
    signer 'anon' and language 'generic' instead of crashing."""
    from app import config
    from app.services import sign_data
    d = tmp_path / "samples" / "Hello"
    d.mkdir(parents=True)
    legacy = {"label": "Hello", "sequence": _pose_seq("Hello", seed=5),
              "recorded_at": "2026-01-01T00:00:00+00:00", "scope": "anon"}
    (d / "legacy.json").write_text(json.dumps(legacy))
    monkeypatch.setattr(config, "SIGN_SAMPLES_DIR", str(tmp_path / "samples"))
    stored = sign_data.load_all_detailed()
    assert len(stored) == 1
    assert stored[0]["signer_id"] == "anon"
    assert stored[0]["language"] == "generic"
    assert sign_data.counts()["total_signers"] == 0  # "anon" is not a signer


# ---------------------------------------------------------------- splits
def test_split_by_signer():
    """Two signers -> whole signer held out; every validated label also trained."""
    from app.ml.training.train import split_samples
    samples = _fixture_samples(["alice", "bob"], per_label=5)
    train, val, split, train_signers, val_signers = split_samples(samples)
    assert split == "by_signer"
    assert len(val_signers) == 1 and len(train_signers) == 1
    assert set(val_signers) | set(train_signers) == {"alice", "bob"}
    assert {it["signer_id"] for it in val} == set(val_signers)
    assert {it["signer_id"] for it in train} == set(train_signers)
    assert {it["label"] for it in val} <= {it["label"] for it in train}
    assert len(val) == 30 and len(train) == 30  # 6 labels x 5 per signer


def test_split_random_single_signer():
    from app.ml.training.train import split_samples
    samples = _fixture_samples(["alice"], per_label=5)
    train, val, split, train_signers, val_signers = split_samples(samples)
    assert split == "random"
    assert len(val) == 6  # 1 held out per label (5 samples -> 20% -> 1)
    assert len(train) == 24
    assert {it["label"] for it in val} == set(LABELS)


def test_split_none_too_few():
    from app.ml.training.train import split_samples
    samples = [{"label": "Hello", "sequence": _pose_seq("Hello", seed=1),
                "signer_id": "alice"},
               {"label": "Yes", "sequence": _pose_seq("Yes", seed=2),
                "signer_id": "alice"}]
    train, val, split, _, _ = split_samples(samples)
    assert split == "none"
    assert val == [] and len(train) == 2


def test_split_by_signer_falls_back_when_label_unique_to_held_out():
    """A label recorded by only the held-out signer cannot be validated
    by-signer (it would have no training samples) -> random split."""
    from app.ml.training.train import split_samples
    samples = _fixture_samples(["alice", "bob"], per_label=5)
    # "Emergency" only from bob.
    samples = [s for s in samples
               if not (s["label"] == "Emergency" and s["signer_id"] == "alice")]
    train, val, split, _, _ = split_samples(samples)
    assert split in ("by_signer", "random")
    if split == "by_signer":
        assert "Emergency" not in {it["label"] for it in val}
    assert len(train) > 0


# ---------------------------------------------------------------- evaluation
def test_compute_metrics_known_values():
    """Hand-computed case: A 1/1 prec, 1/2 rec; B 2/3 prec, 2/2 rec; C perfect."""
    from app.ml.evaluation.evaluate import compute_metrics
    y_true = [0, 0, 1, 1, 2]
    y_pred = [0, 1, 1, 1, 2]
    m = compute_metrics(y_true, y_pred, ["A", "B", "C"])
    assert m["accuracy"] == pytest.approx(0.8)
    assert m["n"] == 5
    assert m["confusion_matrix"]["matrix"] == [[1, 1, 0], [0, 2, 0], [0, 0, 1]]
    a, b, c = m["per_class"]["A"], m["per_class"]["B"], m["per_class"]["C"]
    assert (a["precision"], a["recall"], a["f1"], a["support"]) == (1.0, 0.5, pytest.approx(2 / 3), 2)
    assert b["precision"] == pytest.approx(2 / 3) and b["recall"] == 1.0
    assert b["f1"] == pytest.approx(0.8) and b["support"] == 2
    assert (c["precision"], c["recall"], c["f1"], c["support"]) == (1.0, 1.0, 1.0, 1)


def test_compute_metrics_undefined_is_null_not_zero():
    """A class the model never predicts has precision null, not 0."""
    from app.ml.evaluation.evaluate import compute_metrics
    m = compute_metrics([0, 0, 1], [0, 0, 0], ["A", "B"])
    assert m["per_class"]["B"]["precision"] is None
    assert m["per_class"]["B"]["recall"] == 0.0
    assert m["per_class"]["B"]["f1"] is None
    assert m["per_class"]["B"]["support"] == 1


def test_evaluate_model_on_fixtures_structure():
    """Untrained model on SYNTHETIC FIXTURES: metrics are structural
    only (no claim about accuracy); supports and matrix must add up."""
    from app.ml.evaluation.evaluate import evaluate_model
    from app.ml.models.sign_lstm import SignLSTM
    samples = _fixture_samples(["alice"], per_label=2)
    model = SignLSTM(num_classes=6, hidden_dim=16)
    model.eval()
    report = evaluate_model(model, samples)
    assert set(report["per_class"]) == set(LABELS)
    assert sum(v["support"] for v in report["per_class"].values()) == 12
    matrix = report["confusion_matrix"]["matrix"]
    assert sum(sum(row) for row in matrix) == 12
    assert report["confusion_matrix"]["labels"] == LABELS
    assert report["n"] == 12
    assert 0.0 <= report["accuracy"] <= 1.0


def test_evaluate_model_no_samples_raises():
    from app.ml.evaluation.evaluate import evaluate_model
    from app.ml.models.sign_lstm import SignLSTM
    model = SignLSTM(num_classes=6, hidden_dim=16)
    model.eval()
    with pytest.raises(ValueError, match="No held-out samples"):
        evaluate_model(model, [])


# ---------------------------------------------------------------- train results
def test_train_no_validation_data_reports_val_available_false(tmp_path):
    """Too few samples -> split 'none', val_available False, metrics None.
    Runs a real (tiny) training job on SYNTHETIC FIXTURES."""
    from app.ml.training.train import train_from_samples
    samples = [{"label": "Hello", "sequence": _pose_seq("Hello", seed=1),
                "signer_id": "alice"},
               {"label": "Yes", "sequence": _pose_seq("Yes", seed=2),
                "signer_id": "alice"}]
    out = str(tmp_path / "sign_lstm.pt")
    result = train_from_samples(samples, out, epochs=2, hidden_dim=16)
    assert result["split"] == "none"
    assert result["val_available"] is False
    assert result["val_accuracy"] is None
    assert result["per_class"] is None
    assert result["confusion_matrix"] is None
    assert result["train_samples"] == 2 and result["val_samples"] == 0
    assert os.path.exists(out)


def test_train_by_signer_reports_metrics(tmp_path):
    """Two-signer SYNTHETIC FIXTURES -> by_signer split with a real
    per-class report and a confusion matrix summing to the val size."""
    from app.ml.training.train import train_from_samples
    samples = _fixture_samples(["alice", "bob"], per_label=4)
    out = str(tmp_path / "sign_lstm.pt")
    result = train_from_samples(samples, out, epochs=3, hidden_dim=32)
    assert result["split"] == "by_signer"
    assert result["val_available"] is True
    assert result["val_samples"] == 24  # 6 labels x 4 from the held-out signer
    assert result["train_samples"] == 24
    assert set(result["per_class"]) == set(LABELS)
    for m in result["per_class"].values():
        assert m["support"] == 4
        for k in ("precision", "recall", "f1"):
            assert m[k] is None or 0.0 <= m[k] <= 1.0
    matrix = result["confusion_matrix"]["matrix"]
    assert sum(sum(row) for row in matrix) == 24
    assert 0.0 <= result["val_accuracy"] <= 1.0
    assert len(result["train_signers"]) == 1 and len(result["val_signers"]) == 1
