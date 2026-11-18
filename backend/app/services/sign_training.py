"""Background training jobs for the sign model.

POST /api/ai/sign/train starts a job in a worker thread and returns
immediately (training a torch LSTM must never block the request or the
FastAPI event loop). One job at a time: a second request while a job is
queued/running gets HTTP 409. Jobs are in-memory (lost on restart);
status is polled via GET /api/ai/sign/train/{job_id}.

On success the new weights are saved to SIGN_MODEL_PATH and the cached
inference model is cleared, so the next prediction uses the new model.
"""
import threading
import time
import uuid

from app import config
from app.services import sign_data, sign_model_service
from app.ml.training.train import train_from_samples

_jobs: dict = {}
_jobs_lock = threading.Lock()
_active_lock = threading.Lock()
_active_job_id = None


def _public(job: dict) -> dict:
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "progress": job["progress"],
        "epochs_done": job["epochs_done"],
        "epochs_total": job["epochs_total"],
        "val_accuracy": job["val_accuracy"],
        "message": job["message"],
        # Evaluation report (None/false until a job finishes with a
        # validation set — never fabricated; see train_from_samples).
        "split": job.get("split"),
        "val_available": job.get("val_available", False),
        "per_class": job.get("per_class"),
        "confusion_matrix": job.get("confusion_matrix"),
        "train_signers": job.get("train_signers", []),
        "val_signers": job.get("val_signers", []),
    }


def get_job(job_id: str):
    job = _jobs.get(job_id)
    return _public(job) if job else None


def start_train(epochs: int = 30, hidden_dim: int = 128) -> dict:
    """Start a training job. Raises RuntimeError if one is already active."""
    global _active_job_id
    if not sign_model_service.torch_available():
        raise RuntimeError(
            "PyTorch is not installed, so training cannot run. Install torch (CPU).")
    with _active_lock:
        if _active_job_id is not None:
            job = _jobs.get(_active_job_id)
            if job and job["status"] in ("queued", "running"):
                raise RuntimeError(
                    f"Training job {_active_job_id} is already {job['status']}. "
                    "Wait for it to finish before starting another.")
        job_id = uuid.uuid4().hex[:12]
        job = {
            "job_id": job_id, "status": "queued", "progress": 0.0,
            "epochs_done": 0, "epochs_total": epochs, "val_accuracy": None,
            "message": "Training queued.",
            "split": None, "val_available": False, "per_class": None,
            "confusion_matrix": None, "train_signers": [], "val_signers": [],
        }
        _jobs[job_id] = job
        _active_job_id = job_id
    thread = threading.Thread(
        target=_run_job, args=(job_id, epochs, hidden_dim), daemon=True,
        name=f"sign-train-{job_id}")
    thread.start()
    return _public(job)


def _run_job(job_id: str, epochs: int, hidden_dim: int):
    global _active_job_id
    job = _jobs[job_id]
    try:
        job["status"] = "running"
        job["message"] = "Loading recorded samples…"
        # Detailed form carries signer_id, so the train/val split can
        # hold out a whole signer when >= 2 contributors exist.
        samples = sign_data.load_all_detailed()
        if len(samples) < 2:
            raise ValueError(
                "Not enough recorded samples to train (need at least 2). "
                "Record samples via POST /api/ai/sign/collect.")

        def _progress(epoch_done, epoch_total, loss, val_accuracy):
            job["epochs_done"] = epoch_done
            job["epochs_total"] = epoch_total
            job["progress"] = round(epoch_done / max(epoch_total, 1), 3)
            job["val_accuracy"] = val_accuracy
            job["message"] = (
                f"Epoch {epoch_done}/{epoch_total} loss={loss:.4f}"
                + (f" val_accuracy={val_accuracy:.2%}" if val_accuracy is not None else ""))

        result = train_from_samples(
            samples, config.SIGN_MODEL_PATH, epochs=epochs,
            hidden_dim=hidden_dim, progress_cb=_progress)
        sign_model_service.clear_model_cache()
        job["status"] = "done"
        job["progress"] = 1.0
        job["epochs_done"] = result["epochs"]
        job["val_accuracy"] = result["val_accuracy"]
        job["split"] = result["split"]
        job["val_available"] = result["val_available"]
        job["per_class"] = result["per_class"]
        job["confusion_matrix"] = result["confusion_matrix"]
        job["train_signers"] = result["train_signers"]
        job["val_signers"] = result["val_signers"]
        val_txt = (f"{result['val_accuracy']:.2%}"
                   if result["val_accuracy"] is not None
                   else "n/a (too few samples to hold out a validation set)")
        split_txt = {
            "by_signer": "validated on a held-out signer",
            "random": "validated on a random held-out split",
            "none": "no validation set",
        }[result["split"]]
        job["message"] = (
            f"Training complete. Weights saved to {config.SIGN_MODEL_PATH}. "
            f"train={result['train_samples']} val={result['val_samples']} "
            f"({split_txt}) val_accuracy={val_txt}. "
            "The trained model is now live.")
    except Exception as exc:
        job["status"] = "failed"
        job["message"] = f"Training failed: {exc}"
    finally:
        with _active_lock:
            if _active_job_id == job_id:
                _active_job_id = None
