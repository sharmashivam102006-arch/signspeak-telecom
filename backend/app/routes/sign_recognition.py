"""Sign-recognition API — trained model only by default.

POST /predict uses the trained SignLSTM. If no trained model is
available it answers HTTP 503 {mode: "model-not-trained"} — it never
silently falls back to heuristics. Pass demo=true to explicitly use
the labelled demo pose heuristics instead.
"""
import io

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.deps import get_current_user, get_optional_user
from app.models import User
from app.schemas import (SignCollectRequest, SignPredictRequest,
                         SignPredictResponse, SignTrainRequest, TrainJobOut)
from app.services import sign_data, sign_model_service, sign_training, vision_service
from app.services.sign_model_service import VOCABULARY, ModelNotTrainedError

# Admin gate for training is owned by the security workstream
# (app.deps.get_admin_user: authenticated user whose email is in
# config.ADMIN_EMAILS or id in config.ADMIN_USER_IDS, else 403).
# Until it lands, the import below is None and POST /train fails
# closed with a clear 503 instead of running unauthenticated — this
# route invents no admin logic of its own.
try:
    from app.deps import get_admin_user
except ImportError:  # pragma: no cover - pending security workstream
    get_admin_user = None  # type: ignore[assignment]


def _require_train_admin(user: User = Depends(get_current_user)) -> User:
    """Training is admin-only. Fails closed (503) while the admin
    dependency is not wired yet; delegates to get_admin_user (403 for
    non-admins) once it exists."""
    if get_admin_user is None:
        raise HTTPException(
            status_code=503,
            detail="Training is disabled: admin authorization "
                   "(app.deps.get_admin_user) has not been wired yet. "
                   "Training is admin-only once the security work lands.")
    return get_admin_user(user)


router = APIRouter(prefix="/api/ai/sign", tags=["ai-sign"])


@router.get("/vocabulary")
def vocabulary():
    return {"vocabulary": VOCABULARY, "count": len(VOCABULARY),
            "note": "Initial vocabulary. Extend VOCABULARY in sign_model_service.py "
                    "and retrain SignLSTM to add signs."}


@router.get("/status")
def status():
    """Honest sign-pipeline status: what is installed / trained / recorded."""
    counts = sign_data.counts()
    return {
        "mode": ("trained-model" if sign_model_service.trained_available()
                 else "model-not-trained"),
        "trained": sign_model_service.trained_available(),
        "torch_available": sign_model_service.torch_available(),
        "mediapipe_available": vision_service.mediapipe_available(),
        "model_path": sign_model_service.config.SIGN_MODEL_PATH,
        "samples_per_sign": counts["samples_per_sign"],
        "samples_per_signer": counts["samples_per_signer"],
        "samples_per_sign_by_signer": counts["samples_per_sign_by_signer"],
        "languages": counts["languages"],
        "signers": counts["signers"],
        "total_signers": counts["total_signers"],
        "total_samples": counts["total"],
        "vocabulary": VOCABULARY,
    }


@router.post("/predict", response_model=SignPredictResponse)
async def predict(payload: SignPredictRequest,
                  user: User | None = Depends(get_optional_user)):
    # Duplicate suppression is scoped per caller, so one user's sign
    # never suppresses another user's identical sign.
    try:
        result = await sign_model_service.predict_async(
            landmarks=payload.landmarks or None,
            sequence=payload.sequence or None,
            force=payload.force,
            scope=user.id if user else "anon",
            demo=payload.demo,
        )
    except ModelNotTrainedError as exc:
        raise HTTPException(status_code=503, detail={
            "message": str(exc), "mode": "model-not-trained", "trained": False,
        })
    return SignPredictResponse(
        prediction=result.get("prediction"), confidence=result.get("confidence", 0.0),
        mode=result.get("mode", "trained-model"),
        requires_confirm=result.get("requires_confirm", True),
        message=result.get("message", ""), vocabulary=VOCABULARY,
    )


@router.post("/extract")
async def extract(file: UploadFile = File(...),
                  user: User = Depends(get_current_user)):
    """Extract 21 MediaPipe hand landmarks from an uploaded image.

    Requires authentication (MediaPipe is a shared compute resource).
    Returns {landmarks: 21x3 | null, found: bool, mediapipe_available}.
    503 when MediaPipe is unavailable. Never returns fake landmarks.
    """
    if not vision_service.mediapipe_available():
        raise HTTPException(status_code=503, detail={
            "message": "MediaPipe is not installed; server-side landmark extraction "
                       "is unavailable. Send landmarks from the browser (MediaPipe JS) "
                       "to /api/ai/sign/predict instead.",
            "mediapipe_available": False})
    raw = await file.read(10 * 1024 * 1024 + 1)
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image exceeds the 10MB limit")
    if not raw:
        raise HTTPException(status_code=400, detail="Empty image file")
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode the image "
                            "(send a JPEG or PNG)")
    import asyncio

    try:
        landmarks = await asyncio.to_thread(
            vision_service.extract_hand_landmarks, frame)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:  # e.g. native lib missing (libEGL) in this env
        raise HTTPException(status_code=503, detail={
            "message": f"MediaPipe extraction failed in this environment: {exc}",
            "mediapipe_available": False})
    return {"landmarks": landmarks, "found": landmarks is not None,
            "mediapipe_available": True}


@router.post("/collect", status_code=201)
def collect(payload: SignCollectRequest,
            user: User = Depends(get_current_user)):
    """Store one REAL user-recorded landmark sequence for training.

    Authentication is required: the signer_id is taken from the JWT,
    never from the request body (the schema has no signer_id field).
    Validates the label against the vocabulary and the sequence shape
    (5-30 frames of 21x3 floats); capped at SIGN_SAMPLES_PER_LABEL_MAX
    samples per label (oldest dropped). No synthetic data is stored.
    ``count`` is the label's sample count after this save (the trainer
    UI shows "Saved sample N/10" from it).
    """
    try:
        saved = sign_data.save_sample(
            payload.label, payload.sequence, scope=user.id,
            signer_id=user.id, language=payload.language)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    counts = sign_data.counts()
    return {"saved": True, **saved,
            "count": counts["samples_per_sign"].get(payload.label, 0),
            "counts": counts}


@router.get("/samples")
def samples():
    """Recorded-sample counts per sign (and total)."""
    return sign_data.counts()


@router.post("/train", response_model=TrainJobOut, status_code=202)
def train(payload: SignTrainRequest,
          admin: User = Depends(_require_train_admin)):
    """Start a background SignLSTM training job on the recorded samples.

    Admin-only (see _require_train_admin). Returns immediately with
    the job id; poll GET /train/{job_id}. 409 when a training job is
    already running. On success the new weights become live (the
    inference cache is cleared) and the job result carries the
    evaluation report (split, val_available, per-class precision/
    recall/F1, confusion matrix) computed on the held-out set.
    """
    try:
        return sign_training.start_train(epochs=payload.epochs,
                                         hidden_dim=payload.hidden_dim)
    except RuntimeError as exc:
        msg = str(exc)
        if "already" in msg:
            raise HTTPException(status_code=409, detail=msg)
        raise HTTPException(status_code=503, detail=msg)


@router.get("/train/{job_id}", response_model=TrainJobOut)
def train_status(job_id: str):
    job = sign_training.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown training job {job_id}")
    return job
