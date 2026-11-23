# Training the real SignLSTM

Architecture: `backend/app/ml/models/sign_lstm.py` — `SignLSTM`: input 63
(21 MediaPipe hand landmarks x 3 coords), sequence length 30, hidden 128,
2-layer LSTM, 6-class head (Hello / Yes / No / Help / Thank You / Emergency).
Softmax + confidence. The model is **untrained until you record real samples
and train it** — `GET /health` reports `sign_mode:"model-not-trained"` and
`POST /api/ai/sign/predict` answers 503 until weights exist.

## Dataset format — what actually exists

There is **no bundled dataset and no `.npz` support**. Samples come from two
real sources, both recorded landmarks only (never fabricated):

1. **In-app collection (preferred):** the frontend Train Signs panel records
   2-second MediaPipe sequences and calls `POST /api/ai/sign/collect`.
   Each sample is saved as JSON at
   `SIGN_SAMPLES_DIR/<label>/<uuid>.json` (default
   `backend/data/sign_samples/`) with the label, signer id and the raw
   21x3 landmark sequence. Check counts with `GET /api/ai/sign/samples`.
2. **CSV import (CLI only):** one row per frame, frames grouped by sample:
   `sample_id,label,frame,landmark_0_x,landmark_0_y,landmark_0_z,...,landmark_20_z`
   — see `backend/app/ml/training/train.py --help`.

Each sample is 5–30 frames of 21 `[x, y, z]` landmarks. Normalisation is
server-side (`backend/app/ml/preprocessing/normalize.py`): wrist-centred,
scaled by wrist→middle-MCP distance, then flattened to 63 floats per frame
and padded/truncated to 30 frames — so the model is position/scale
invariant. Record **at least 5 samples per sign, from at least 2–3 different
signers**, or the model will not generalise; training refuses to run with
samples for fewer than 2 different signs.

## Train

Via the API (what the UI does):

```bash
# after recording samples for at least 2 signs:
curl -X POST http://localhost:8000/api/ai/sign/train \
  -H "Authorization: Bearer <JWT>" -H "Content-Type: application/json" -d '{}'
# poll the job:
curl http://localhost:8000/api/ai/sign/train/<job_id> \
  -H "Authorization: Bearer <JWT>"
```

Via the CLI (same training code, CSV input):

```bash
cd backend
# CPU torch wheel — plain `pip install torch` pulls the CUDA wheel:
pip install --index-url https://download.pytorch.org/whl/cpu torch
python -m app.ml.training.train --data data/signs.csv --epochs 30 --out app/ml/models/sign_lstm.pt
```

With no `--data`, the script prints these instructions and exits 0 — it
never invents training data.

## How training is evaluated (honestly)

`train_from_samples()` (used by both entry points) splits samples
**by signer first** when ≥2 signers exist (one signer's samples held out
for validation, so the metrics measure generalisation to a new signer),
falling back to a stratified random split (~20% per label) and then to no
validation set. The split used is recorded in the job result. Every train
job with validation data reports overall `val_accuracy` **plus per-class
precision, recall, F1 and support and a confusion matrix**
(`GET /api/ai/sign/train/{job_id}` → `per_class`, `confusion_matrix`,
`val_available`). `val_accuracy` is `null` (and `val_available: false`)
when no validation data exists — the API reports that as null instead of
inventing a number. `python -m app.ml.evaluation.evaluate` can re-evaluate
saved weights against held-out samples at any time. Do not quote a
per-sign accuracy until a real evaluation on held-out signers exists.

## Serve

Weights are written to the path in `SIGN_MODEL_PATH`
(default, relative to `backend/`: `app/ml/models/sign_lstm.pt` —
**not** `models/sign_lstm.pt`). Restart the backend; `GET /health` then
reports `sign_mode:"trained-model"` and `POST /api/ai/sign/predict` serves
the model (same response shape; confidence is now a real softmax score and
can exceed 0.85). No frontend change is needed. To add a sign: extend
`VOCABULARY` in `app/ml/models/sign_lstm.py` **and** the copy in
`app/services/sign_model_service.py` (both lists must match), collect data,
retrain.
