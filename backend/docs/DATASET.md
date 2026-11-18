# SignSpeak Telecom — Sign-Language Dataset

This document is the single source of truth for the sign-recognition
dataset: what the vocabulary is, what language it belongs to, how
samples are collected, and how the first trained model is produced.
It is updated whenever the dataset status changes.

## Current status (17 November 2026)

| Item | Status |
|---|---|
| Recorded samples | **0** |
| Distinct signers | **0** |
| Trained model weights | **None** — `app/ml/models/sign_lstm.pt` does not exist |
| Live recognition | **Not available** — `POST /api/ai/sign/predict` answers HTTP 503 `model-not-trained` until weights exist |
| Training pipeline | Proven on synthetic test fixtures only (labelled as fixtures in `backend/tests/`); never run on real data |

Nothing in this document claims a sign is recognised. Recognition
begins only after real humans record real samples and a model is
trained and evaluated on held-out data.

## Vocabulary

Six introductory signs, fixed in `app/services/sign_model_service.py`
(`VOCABULARY`):

| Sign | Meaning |
|---|---|
| Hello | Greeting |
| Yes | Affirmation |
| No | Negation |
| Help | Request for assistance |
| Thank You | Gratitude |
| Emergency | Urgent help needed |

To add a sign: extend `VOCABULARY`, record samples for it, retrain.
The model has one output class per vocabulary entry.

## Language position — read this before contributing

These six signs are **generic introductory hand signs**. They are
**NOT certified Indian Sign Language (ISL)**, and this project does
not claim otherwise. Do not mix sign languages:

- Every recorded sample stores a `language` label (default
  `"generic"`). A contributor may set `"ISL"` **only** when they
  confirm the performed sign is Indian Sign Language.
- Samples with different `language` labels must not be mixed into
  one training run without recording that fact in the training job
  notes and in this document.
- The demo pose heuristics (`demo=true`) recognise simple hand poses
  (open palm, fist, …), not sign language of any kind. They are
  labelled `demo-rule-based`, capped at 0.78 confidence, and are
  never used by the trained-model path.

## What a sample is

One sample = one 2-second recording of a single sign, stored as a
JSON file `<SIGN_SAMPLES_DIR>/<label>/<uuid>.json`:

```json
{
  "label": "Hello",
  "sequence": [[[x, y, z], ... 21 landmarks], ... 5-30 frames],
  "recorded_at": "2026-11-17T12:00:00+00:00",
  "scope": "<signer user id>",
  "signer_id": "<authenticated user id — set server-side from the JWT>",
  "language": "generic"
}
```

- Landmarks are the 21 MediaPipe hand landmarks per frame, extracted
  in the browser by `@mediapipe/tasks-vision` (or server-side by
  `POST /api/ai/sign/extract`). Frames are normalised at train/inference
  time (wrist to origin, scale by wrist→middle-MCP distance; see
  `app/ml/preprocessing/normalize.py`).
- Sequences are 5–30 frames; the trainer records 2 seconds at ~10 fps
  (≈20 frames) and the model consumes a 30-frame window
  (shorter sequences are zero-padded at the front).
- `signer_id` is **never** taken from the request body. The collect
  endpoint requires authentication and sets it from the JWT.
- Per-label cap: `SIGN_SAMPLES_PER_LABEL_MAX` (default 200); oldest
  samples are dropped when the cap is exceeded.

## Collection protocol

Contributors use the **Train Signs** panel in a call
(`frontend/components/SignTrainer.tsx`):

1. Sign in (recording requires an account — anonymous samples are
   not accepted, so every sample is attributable to a signer).
2. Pick a sign, hold your hand in view of the camera, and record a
   2-second sample. The panel discards the recording if the hand was
   visible for less than 70% of the 2 seconds or fewer than 5 frames
   were captured.
3. Repeat. **Target: ≥ 10 samples per sign, per signer.**
4. **Multiple signers are required for honest validation.** Aim for
   at least 2–3 signers, each recording all six signs. A model
   trained on one person's hands has not been shown to recognise
   anyone else's.
5. Record signs naturally and varied: different hand positions,
   speeds, and (where safe) different backgrounds/lighting. Do not
   record the same frozen pose repeatedly — variation is the data.

Minimums enforced by the training code: at least 2 samples total,
covering at least 2 different signs, or training refuses to start.
Useful minimums for a first real model: ≥ 5 samples per sign
(training unlocks in the UI at 5), ideally ≥ 10 per sign from ≥ 2
signers so a whole signer can be held out for validation.

## Training and validation

Start training (admin-only) with `POST /api/ai/sign/train`
(`{epochs, hidden_dim}`; defaults 30 / 128) and poll
`GET /api/ai/sign/train/{job_id}`. Training runs in a background
thread; one job at a time (a second request gets 409).

### Train/validation split

The split is chosen by `app/ml/training/train.py::split_samples`
and recorded in the job result (`split`):

| Split | When | What it means |
|---|---|---|
| `by_signer` | Samples from ≥ 2 signers, and every validated label also has training samples | One whole signer is held out for validation. This is the honest test: it measures recognition of a person the model never trained on. |
| `random` | Single signer (or the by-signer split is impossible) | Stratified per label: ~20% held out (≥ 1 sample when a label has ≥ 3 samples). Weaker evidence — the model has seen the same hands in training. |
| `none` | Too few samples to hold any out | No validation data. The job result says `val_available: false` and all metrics are null. Never read accuracy into this case. |

### Evaluation report

When a validation set exists, the job result carries the full
evaluation (computed by `app/ml/evaluation/evaluate.py`):

```json
{
  "split": "by_signer",
  "val_available": true,
  "val_accuracy": 0.83,
  "per_class": {
    "Hello": {"precision": 0.9, "recall": 0.8, "f1": 0.85, "support": 10}
  },
  "confusion_matrix": {"labels": ["Hello", "Yes", "No", "Help", "Thank You", "Emergency"],
                       "matrix": [[8, 1, 0, ...], ...]},
  "train_signers": ["<user id A>"],
  "val_signers": ["<user id B>"]
}
```

- `per_class` — precision, recall, F1 and support per sign. A metric
  is `null` when undefined for that class (e.g. precision when the
  model never predicted the class); null is reported, never replaced.
- `confusion_matrix` — rows are true labels, columns predicted labels.
- Evaluate a weights file independently any time with:
  `python -m app.ml.evaluation.evaluate --weights <path> [--samples-dir <dir>]`

### Thresholds and duplicate suppression

Inference confidence thresholds come from the server environment
(`SIGN_THRESHOLD_HIGH=0.85`, `SIGN_THRESHOLD_LOW=0.60`):

- ≥ 0.85 — high confidence: prediction is shown directly.
- 0.60–0.85 — medium: the UI asks the user to confirm.
- < 0.60 — low: the UI asks the user to repeat or correct the sign.

Thresholds are **configuration, not measured accuracy**. A threshold
says when the UI trusts a score; only the held-out per-class metrics
above say how accurate the model is. The same prediction within 2
seconds from the same caller is suppressed as a duplicate (per-caller
scope, so one user's sign never suppresses another's); see
`app/services/sign_model_service.py`.

## Steps to produce the first trained model

1. **Record samples.** At least 2 signers × 6 signs × 5 samples
   (target 10). Check progress at `GET /api/ai/sign/status` — it
   reports `samples_per_sign`, `samples_per_signer` and
   `total_signers`.
2. **Train.** As an admin, `POST /api/ai/sign/train`. Poll the job.
   Confirm the result says `split: "by_signer"` (or `"random"` if
   only one signer contributed) and `val_available: true`.
3. **Evaluate.** Read `per_class` and the confusion matrix in the
   job result. Every sign should have precision/recall/F1 reported
   with real support. If a sign scores poorly, record more varied
   samples for that sign and retrain — do not lower the thresholds
   to hide it.
4. **Ship the weights.** On success the weights are saved to
   `SIGN_MODEL_PATH` (`app/ml/models/sign_lstm.pt`) and the inference
   cache is cleared, so the next prediction uses the new model.
   `GET /api/ai/sign/status` should then report `"trained": true`.
5. **Verify live.** Perform each sign in front of a camera and
   confirm `POST /api/ai/sign/predict` returns `mode:
   "trained-model"` with genuine softmax confidences. Record the
   observed per-sign results here before claiming recognition works.
6. **Update this document** with the sample counts, signers, split,
   and per-class metrics of the shipped model.

## What this pipeline does NOT do

- It does not invent training samples. No synthetic data is ever
  written to the sample store by production code.
- It does not train on random or jittered fixtures and ship those
  weights. Fixtures exist only in `backend/tests/` and are labelled
  as fixtures.
- It does not claim recognition accuracy from configurable thresholds.
- It does not recognise continuous signing, two-handed signs, or
  signs outside the six-word vocabulary. The model sees one hand,
  21 landmarks, 30 frames.
