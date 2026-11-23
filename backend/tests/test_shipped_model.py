"""Tests for the trained sign model shipped at app/ml/models/sign_lstm.pt
(10 signs, trained on WLASL — see the training report).

* The sign list must be identical, in the same order, in the backend
  service, the model class module, the frontend, and the checkpoint
  itself: output class i of the model is VOCABULARY[i].
* With PyTorch installed (CI), the real checkpoint must load and serve a
  "trained-model" prediction, and a checkpoint with a different sign
  order must be refused rather than silently mislabelling signs.

Run: python -m pytest tests/test_shipped_model.py  (from backend/)
"""
import os
import pickletools
import re
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(BACKEND, "app", "ml", "models", "sign_lstm.pt")
STORE_TS = os.path.join(BACKEND, "..", "frontend", "lib", "store.ts")


def _checkpoint_vocabulary(path: str) -> list:
    """Read the 'vocabulary' list from a torch checkpoint WITHOUT
    unpickling it: parse the pickle opcodes and take the strings that
    follow the 'vocabulary' key."""
    with zipfile.ZipFile(path) as z:
        data = z.read(next(n for n in z.namelist() if n.endswith("data.pkl")))
    strings = [arg for op, arg, _ in pickletools.genops(data)
               if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE")]
    return strings[strings.index("vocabulary") + 1:]


def _source_vocabulary(path: str, name: str) -> list:
    src = open(path, encoding="utf-8").read()
    m = re.search(name + r"\s*=\s*\[(.*?)\]", src, re.S)
    assert m, f"{name} not found in {path}"
    return re.findall(r'"([^"]+)"', m.group(1))


def test_vocabulary_lists_agree_everywhere():
    from app.services.sign_model_service import VOCABULARY
    assert len(VOCABULARY) == 10
    model_py = os.path.join(BACKEND, "app", "ml", "models", "sign_lstm.py")
    assert _source_vocabulary(model_py, "VOCABULARY") == VOCABULARY
    if os.path.exists(STORE_TS):  # backend-only checkouts have no frontend
        assert _source_vocabulary(STORE_TS, "SIGN_VOCABULARY") == VOCABULARY


def test_shipped_checkpoint_matches_vocabulary():
    from app.services.sign_model_service import VOCABULARY
    assert os.path.exists(MODEL), "trained model missing from app/ml/models/"
    assert _checkpoint_vocabulary(MODEL) == VOCABULARY


def _open_palm():
    lm = [[0.5, 0.9, 0.0] for _ in range(21)]
    lm[9] = [0.5, 0.7, 0.0]
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        lm[pip] = [0.5, 0.6, 0.0]
        lm[tip] = [0.5, 0.4, 0.0]
    return lm


def test_shipped_model_loads_and_predicts(monkeypatch):
    pytest.importorskip("torch")
    from app import config
    from app.services import sign_model_service as svc
    monkeypatch.setattr(config, "SIGN_MODEL_PATH", MODEL)
    svc.clear_model_cache()
    try:
        assert svc.trained_available()
        out = svc.predict(sequence=[_open_palm() for _ in range(30)], force=True,
                          scope="test-shipped-model")
        assert out["mode"] == "trained-model"
        assert out["prediction"] in svc.VOCABULARY
        assert 0.0 < out["confidence"] <= 1.0
    finally:
        svc.clear_model_cache()


def test_checkpoint_with_other_sign_order_is_refused(monkeypatch):
    pytest.importorskip("torch")
    from app import config
    from app.services import sign_model_service as svc
    monkeypatch.setattr(config, "SIGN_MODEL_PATH", MODEL)
    swapped = list(svc.VOCABULARY)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    monkeypatch.setattr(svc, "VOCABULARY", swapped)
    svc.clear_model_cache()
    try:
        with pytest.raises(svc.ModelNotTrainedError, match="trained on signs"):
            svc._load_model()
    finally:
        svc.clear_model_cache()
