"""Sign-recognition service — REAL trained model only by default.

Paths, never confused:
  * "trained-model" — PyTorch SignLSTM loaded from SIGN_MODEL_PATH.
    This is the ONLY path used by POST /api/ai/sign/predict unless the
    caller explicitly sends ``demo=true``. If torch is missing or no
    weights exist, predict() raises :class:`ModelNotTrainedError` and
    the route answers HTTP 503 {mode: "model-not-trained"} — it never
    silently falls back to a heuristic and presents it as recognition.
  * "demo-rule-based" — deterministic hand-pose heuristics on the 21
    MediaPipe landmarks (see _demo_predict). ONLY when the caller
    explicitly asks (demo=true). EXPLICITLY A DEMO: it recognises simple
    hand poses, not real sign language, and its confidence is capped at
    0.78 so it always requires confirmation (threshold high = 0.85).
    Never label its output as a trained model.

Prediction smoothing / duplicate suppression: the same prediction
within 2 seconds is suppressed (returns the previous result with
duplicate=True) unless force=True.
Thresholds from config: HIGH=0.85 (auto-accept), LOW=0.60 (ask to repeat).

Inference runs in a threadpool (predict_async) so the FastAPI event
loop is never blocked by torch. The model is a lazy singleton: loaded
once on first use, never at import/startup.
"""
import asyncio
import os
import threading
import time
from typing import Dict, List, Optional, Tuple

from app import config
from app.ml.preprocessing.normalize import normalize_landmarks, sequence_to_features
from app.services.landmark_service import landmarks_valid

# Order matters: index i is output class i of the trained model
# (backend/app/ml/models/sign_lstm.pt, trained on WLASL, 10 signs).
VOCABULARY = ["Hello", "Yes", "No", "Help", "Thank You", "Emergency",
              "Please", "Sorry", "Good", "Goodbye"]
SEQUENCE_LENGTH = 30
FEATURE_DIM = 63  # 21 landmarks * 3 coords

# Duplicate suppression is scoped per caller (user id or "anon") — a global
# would suppress one user's sign because another user just signed it.
_last_prediction_by_scope: Dict[str, Tuple[Optional[str], float]] = {}
_SUPPRESS_SECONDS = 2.0

# Trained-model cache: load weights once, not once per request.
_cached_model = None
_cached_model_path: Optional[str] = None
_model_lock = threading.Lock()


class ModelNotTrainedError(RuntimeError):
    """No trained sign model is available (torch missing or no weights).

    The API turns this into HTTP 503 with mode="model-not-trained".
    Record samples via POST /api/ai/sign/collect, then train with
    POST /api/ai/sign/train — or set SIGN_MODEL_PATH to a weights file.
    """


def torch_available() -> bool:
    try:
        import torch  # type: ignore  # noqa: F401
        return True
    except ImportError:
        return False


def weights_available() -> bool:
    return bool(config.SIGN_MODEL_PATH) and os.path.exists(config.SIGN_MODEL_PATH)


def trained_available() -> bool:
    """True only when torch is installed AND a weights file exists."""
    return torch_available() and weights_available()


def clear_model_cache() -> None:
    """Drop the cached model (e.g. after training new weights)."""
    global _cached_model, _cached_model_path
    with _model_lock:
        _cached_model = None
        _cached_model_path = None


def _finger_extended(lm, tip: int, pip: int) -> bool:
    """Finger extended if tip is above (smaller y than) its PIP joint."""
    return lm[tip][1] < lm[pip][1]


def _demo_predict(landmarks: List[List[float]]):
    """Deterministic pose heuristics — DEMO ONLY, not sign language.

    Landmark indices (MediaPipe Hands): 0 wrist, 4 thumb tip, 8 index tip,
    12 middle tip, 16 ring tip, 20 pinky tip; PIPs 6/10/14/18.
    Heuristics:
      open palm (all 4 fingers extended)        -> Hello (0.72)
      index only extended                       -> Yes   (0.68)
      fist (no fingers extended)                -> No    (0.66)
      thumb extended only / thumbs-up-ish       -> Thank You (0.62)
      index+middle extended (V)                 -> Help  (0.60)
      pinky only extended                       -> Emergency (0.55)
    Anything else -> no confident pose.
    """
    lm = normalize_landmarks(landmarks)
    idx = _finger_extended(lm, 8, 6)
    mid = _finger_extended(lm, 12, 10)
    ring = _finger_extended(lm, 16, 14)
    pinky = _finger_extended(lm, 20, 18)
    thumb_up = lm[4][1] < lm[3][1]  # thumb tip above thumb IP
    extended = sum([idx, mid, ring, pinky])
    if extended == 4:
        return "Hello", 0.72
    if idx and not mid and not ring and not pinky:
        return "Yes", 0.68
    if extended == 0 and not thumb_up:
        return "No", 0.66
    if thumb_up and extended == 0:
        return "Thank You", 0.62
    if idx and mid and not ring and not pinky:
        return "Help", 0.60
    if pinky and not idx and not mid and not ring:
        return "Emergency", 0.55
    return None, 0.0


def _load_model():
    """Load (once) and return the trained SignLSTM, or raise.

    Raises ModelNotTrainedError when torch is missing, the weights file
    is absent, or the file cannot be loaded as a SignLSTM state dict.
    """
    global _cached_model, _cached_model_path
    with _model_lock:
        if _cached_model is not None and _cached_model_path == config.SIGN_MODEL_PATH:
            return _cached_model
        if not torch_available():
            raise ModelNotTrainedError(
                "PyTorch is not installed, so the trained sign model cannot run. "
                "Install torch (CPU) to enable sign recognition.")
        if not weights_available():
            raise ModelNotTrainedError(
                "No trained sign model weights found at SIGN_MODEL_PATH "
                f"({config.SIGN_MODEL_PATH}). Record samples via "
                "POST /api/ai/sign/collect and train with POST /api/ai/sign/train.")
        import torch  # type: ignore
        from app.ml.models.sign_lstm import SignLSTM

        try:
            checkpoint = torch.load(config.SIGN_MODEL_PATH, map_location="cpu",
                                    weights_only=True)
        except Exception as exc:
            raise ModelNotTrainedError(
                f"Sign model weights at {config.SIGN_MODEL_PATH} could not be "
                f"loaded: {exc}. Retrain with POST /api/ai/sign/train.") from exc
        # Checkpoints saved by train_from_samples carry the architecture
        # they were trained with; bare state dicts (older files) default
        # to the standard hidden_dim=128.
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
            hidden_dim = int(checkpoint.get("hidden_dim", 128))
            # A checkpoint that records its sign list must match ours in
            # the same order: output class i is VOCABULARY[i]. A different
            # order would load fine and silently mislabel every sign.
            saved_vocab = checkpoint.get("vocabulary")
            if saved_vocab is not None and list(saved_vocab) != list(VOCABULARY):
                raise ModelNotTrainedError(
                    f"Sign model at {config.SIGN_MODEL_PATH} was trained on signs "
                    f"{list(saved_vocab)}, but the app's vocabulary is {VOCABULARY}. "
                    "Update VOCABULARY (same order) or retrain.")
        else:
            state = checkpoint
            hidden_dim = 128
        model = SignLSTM(input_dim=FEATURE_DIM, num_classes=len(VOCABULARY),
                         hidden_dim=hidden_dim)
        try:
            model.load_state_dict(state)
        except Exception as exc:
            raise ModelNotTrainedError(
                f"Sign model weights at {config.SIGN_MODEL_PATH} could not be "
                f"loaded as a SignLSTM state dict: {exc}. Retrain with "
                "POST /api/ai/sign/train.") from exc
        model.eval()
        _cached_model, _cached_model_path = model, config.SIGN_MODEL_PATH
        return model


def _trained_predict(sequence):
    """Run the trained PyTorch model. Returns (label, confidence).

    Raises ModelNotTrainedError when no trained model is available.
    """
    # _load_model first: it raises ModelNotTrainedError (-> 503) when there
    # are no weights, before anything needs torch to be installed.
    model = _load_model()
    import torch  # type: ignore

    with torch.no_grad():
        # Normalised 30x63 features, exactly as in training.
        feats = sequence_to_features(sequence, seq_len=SEQUENCE_LENGTH)
        x = torch.tensor([feats], dtype=torch.float32)
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)[0]
        idx = int(probs.argmax())
        return VOCABULARY[idx], float(probs[idx])


def _no_hand() -> dict:
    return {
        "prediction": None, "confidence": 0.0, "mode": "no-hand",
        "requires_confirm": True,
        "message": "No valid hand landmarks (need 21 [x,y,z] points).",
        "vocabulary": VOCABULARY,
    }


def _finish(prediction, confidence, mode, scope, force, now) -> dict:
    """Apply duplicate suppression + thresholds to a prediction."""
    if mode == "demo-rule-based":
        confidence = min(confidence, 0.78)
    last_pred, last_time = _last_prediction_by_scope.get(scope, (None, 0.0))
    if (not force and last_pred == prediction
            and now - last_time < _SUPPRESS_SECONDS):
        return {
            "prediction": prediction, "confidence": round(confidence, 3),
            "mode": mode, "requires_confirm": confidence < config.SIGN_THRESHOLD_HIGH,
            "message": f"Duplicate '{prediction}' suppressed (same sign within 2s).",
            "vocabulary": VOCABULARY, "duplicate": True,
        }
    _last_prediction_by_scope[scope] = (prediction, now)
    requires_confirm = confidence < config.SIGN_THRESHOLD_HIGH
    if confidence < config.SIGN_THRESHOLD_LOW:
        msg = "Confidence below 0.60 — please repeat the sign or correct it."
    elif requires_confirm:
        msg = f"Uncertain ({confidence:.0%}) — please confirm '{prediction}'."
    else:
        msg = f"Recognized '{prediction}'."
    if mode == "demo-rule-based":
        msg += " [DEMO rule-based pose heuristic — not a trained sign-language model.]"
    return {
        "prediction": prediction, "confidence": round(confidence, 3), "mode": mode,
        "requires_confirm": requires_confirm, "message": msg, "vocabulary": VOCABULARY,
    }


def predict(landmarks=None, sequence=None, force: bool = False,
            scope: str = "anon", demo: bool = False) -> dict:
    """Predict a sign from landmarks (one frame) or a sequence of frames.

    Default (demo=False): trained model only. Requires a sequence of
    >= 5 frames; raises ModelNotTrainedError when no trained model is
    available. With demo=True the rule-based heuristics run instead and
    the result is labelled demo-rule-based (confidence capped at 0.78).

    A single frame with no sequence is not enough for the temporal
    model: without demo=True it raises ModelNotTrainedError with a
    message explaining that a sequence is needed.
    """
    now = time.time()
    frame = None
    if sequence:
        frame = sequence[-1]
    elif landmarks:
        frame = landmarks
    if frame is None or not landmarks_valid(frame):
        return _no_hand()

    if demo:
        pred, conf = _demo_predict(frame)
        if pred is None:
            return {
                "prediction": None, "confidence": 0.0, "mode": "demo-rule-based",
                "requires_confirm": True,
                "message": "Hand detected but pose did not match a known demo sign. Try again.",
                "vocabulary": VOCABULARY,
            }
        return _finish(pred, conf, "demo-rule-based", scope, force, now)

    # Trained-model path: needs a temporal sequence.
    if not sequence or len(sequence) < 5:
        raise ModelNotTrainedError(
            "The trained sign model needs a sequence of at least 5 frames "
            "(21 landmarks each). Send 'sequence' with 5-30 frames, or pass "
            "demo=true to use the labelled demo pose heuristics instead.")
    prediction, confidence = _trained_predict(sequence)
    return _finish(prediction, confidence, "trained-model", scope, force, now)


async def predict_async(landmarks=None, sequence=None, force: bool = False,
                        scope: str = "anon", demo: bool = False) -> dict:
    """Async wrapper: torch inference runs in a threadpool, never on the
    event loop. ModelNotTrainedError propagates to the caller."""
    return await asyncio.to_thread(
        predict, landmarks, sequence, force, scope, demo)
