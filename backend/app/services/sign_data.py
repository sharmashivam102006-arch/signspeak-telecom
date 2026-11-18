"""Storage for user-recorded sign landmark sequences.

POST /api/ai/sign/collect stores REAL landmark sequences recorded by a
user (the frontend sends MediaPipe landmarks). No synthetic data is
ever written here by production code — tests that need fixtures use
their own temporary directory via SIGN_SAMPLES_DIR.

Layout: <SIGN_SAMPLES_DIR>/<label>/<uuid>.json
Each file: {"label": str, "sequence": [[21x[x,y,z]] ...],
            "recorded_at": iso, "scope": str,
            "signer_id": str, "language": str}
  * signer_id — the authenticated user who recorded the sample. Set
    server-side from the JWT; the client cannot choose it. Samples
    recorded before signer tracking exist get "anon".
  * language — the sign-language label the contributor confirmed for
    this sample, e.g. "generic" (default: introductory hand signs, NOT
    certified Indian Sign Language) or "ISL" (only when the contributor
    confirms the sign is Indian Sign Language). Never mix languages in
    one training run without saying so; see backend/docs/DATASET.md.
Cap: SIGN_SAMPLES_PER_LABEL_MAX files per label (oldest dropped).
"""
import json
import os
import uuid
from collections import Counter
from datetime import datetime, timezone

from app import config
from app.services.sign_model_service import VOCABULARY


def _label_dir(label: str) -> str:
    return os.path.join(config.SIGN_SAMPLES_DIR, label)


def _clean_language(language) -> str:
    """Normalise a language/variant label; default "generic".

    "generic" = introductory hand signs, NOT a certified sign language.
    Contributors may pass "ISL" only when they confirm the sign is
    Indian Sign Language. The value is stored verbatim (lower-cased,
    max 30 chars) so training runs can filter or report on it.
    """
    lang = str(language or "generic").strip().lower() or "generic"
    return lang[:30]


def save_sample(label: str, sequence, scope: str = "anon",
                signer_id: str | None = None, language: str = "generic") -> dict:
    """Validate and store one sequence.

    Returns {label, sample_id, path, signer_id, language}.
    ``signer_id`` must come from the authenticated user server-side —
    the collect route sets it from the JWT and never trusts a
    client-supplied value. ``scope`` is kept for backward compatibility
    and defaults to the signer id.
    """
    if label not in VOCABULARY:
        raise ValueError(f"Unknown label {label!r}; vocabulary is {VOCABULARY}")
    if not isinstance(sequence, list) or not 5 <= len(sequence) <= 30:
        raise ValueError("sequence must have 5-30 frames")
    for frame in sequence:
        if not isinstance(frame, list) or len(frame) != 21:
            raise ValueError("each frame must have 21 landmarks")
        for lm in frame:
            if not isinstance(lm, (list, tuple)) or len(lm) < 3:
                raise ValueError("each landmark must be [x, y, z]")
            float(lm[0]); float(lm[1]); float(lm[2])
    signer = str(signer_id) if signer_id else "anon"
    lang = _clean_language(language)
    d = _label_dir(label)
    os.makedirs(d, exist_ok=True)
    sample_id = uuid.uuid4().hex
    payload = {
        "label": label,
        "sequence": [[[float(v) for v in lm] for lm in frame] for frame in sequence],
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "scope": scope if scope != "anon" else signer,
        "signer_id": signer,
        "language": lang,
    }
    path = os.path.join(d, f"{sample_id}.json")
    with open(path, "w") as f:
        json.dump(payload, f)
    # Enforce the per-label cap: drop oldest files first.
    files = sorted(
        (os.path.join(d, fn) for fn in os.listdir(d) if fn.endswith(".json")),
        key=os.path.getmtime)
    while len(files) > config.SIGN_SAMPLES_PER_LABEL_MAX:
        os.remove(files.pop(0))
    return {"label": label, "sample_id": sample_id, "path": path,
            "signer_id": signer, "language": lang}


def _iter_samples():
    """Yield (label, payload) for every stored sample file.

    Corrupt/unreadable files are skipped, never crash the caller.
    Payloads written before signer/language tracking get defaults
    ("anon" / "generic") so old samples stay usable.
    """
    for label in VOCABULARY:
        d = _label_dir(label)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".json"):
                continue
            try:
                with open(os.path.join(d, fn)) as f:
                    payload = json.load(f)
                if payload.get("label") == label and isinstance(payload.get("sequence"), list):
                    payload.setdefault("signer_id", payload.get("scope") or "anon")
                    payload.setdefault("language", "generic")
                    yield label, payload
            except (OSError, ValueError):
                continue


def counts() -> dict:
    """Sample counts: per label, per signer, per label-per-signer, per language.

    Keys (backward compatible — samples_per_sign/total unchanged):
      samples_per_sign: {label: n}
      samples_per_signer: {signer_id: n}
      samples_per_sign_by_signer: {label: {signer_id: n}}
      languages: {language: n}
      signers: [signer_id, ...] (sorted)
      total_signers: int (distinct signers, "anon" excluded)
      total: int
    """
    per_label = {label: 0 for label in VOCABULARY}
    per_signer: Counter = Counter()
    per_label_signer: dict = {label: Counter() for label in VOCABULARY}
    languages: Counter = Counter()
    total = 0
    for label, payload in _iter_samples():
        signer = str(payload.get("signer_id") or "anon")
        per_label[label] += 1
        per_signer[signer] += 1
        per_label_signer[label][signer] += 1
        languages[_clean_language(payload.get("language"))] += 1
        total += 1
    signers = sorted(per_signer)
    real_signers = [s for s in signers if s != "anon"]
    return {
        "samples_per_sign": per_label,
        "samples_per_signer": dict(per_signer),
        "samples_per_sign_by_signer": {
            label: dict(per_label_signer[label]) for label in VOCABULARY},
        "languages": dict(languages),
        "signers": signers,
        "total_signers": len(real_signers),
        "total": total,
    }


def load_all() -> list:
    """Load all stored samples as [(label, sequence), ...].

    Kept for backward compatibility (training entry point, older tests).
    Use load_all_detailed() when signer/language metadata is needed.
    """
    return [(label, payload["sequence"])
            for label, payload in _iter_samples()]


def load_all_detailed() -> list:
    """Load all stored samples with metadata.

    Returns [{label, sequence, signer_id, language, recorded_at,
    sample_id}, ...] — the form the by-signer train/val split and the
    evaluation report consume. sample_id is the file stem.
    """
    out = []
    for label in VOCABULARY:
        d = _label_dir(label)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".json"):
                continue
            try:
                with open(os.path.join(d, fn)) as f:
                    payload = json.load(f)
                if payload.get("label") == label and isinstance(payload.get("sequence"), list):
                    out.append({
                        "label": label,
                        "sequence": payload["sequence"],
                        "signer_id": str(payload.get("signer_id")
                                         or payload.get("scope") or "anon"),
                        "language": _clean_language(payload.get("language")),
                        "recorded_at": payload.get("recorded_at"),
                        "sample_id": fn[:-5],
                    })
            except (OSError, ValueError):
                continue  # skip a corrupt sample file, never crash training
    return out
