# SignSpeak Telecom

An accessibility layer for live calls. A sign-language user signs; the hearing
participant hears it. The hearing participant speaks; the sign user reads it.
One conversation, two ways of talking.

**What this version actually does:** recognises a small, fixed vocabulary of
isolated signs and speaks them via templates, and turns speech into live
captions. It is not a sign-language translator, and it is not fit for medical
or legal use. That distinction is load-bearing — see
[`docs/critique.md`](docs/critique.md).

---

## Start here: run the demo in 30 seconds

No install, no accounts, no model training.

```bash
cd signspeak
python3 -m http.server 8080
# open http://localhost:8080/demo/
```

Start the camera, go to **Teach signs**, hold `Space` and perform a sign ~10
times, do the same for a second sign and for **NONE**, then sign at the camera.
It will speak. The demo carries the real feature encoder, the real segmenter
and a real confidence/abstention policy — the only thing swapped out is the
trained model, replaced by few-shot DTW k-NN so it works before you own a
dataset.

Serve it over http. Opening the file directly fails: `getUserMedia` needs a
secure context.

---

## Repository map

```
demo/index.html          zero-setup single-file demo — start here
ml/                      research code: features, segmentation, models, calibration, eval
apps/api/                FastAPI control plane: rooms, tokens, transcript, telemetry
apps/web/                Next.js client: call UI + in-browser sign recognition
docs/critique.md         hard read of the original plan — read this second
docs/architecture.md     layer decisions, data flow, latency budget
docs/evaluation-protocol.md   what to measure and why
```

Verify the build at any time:

```bash
make verify     # end-to-end plumbing on synthetic data + Python/TS encoder parity
```

Both are green in this repo: 30 pipeline checks pass, and the Python and
TypeScript feature encoders agree to 2.5e-17 across 24 fixture inputs.

The four files that carry the actual ideas:

| file | what it decides |
| --- | --- |
| `ml/signspeak_ml/features.py` | body-anchored landmark normalisation |
| `ml/signspeak_ml/segment.py` | **when** a sign happened — the hardest piece |
| `ml/signspeak_ml/calibrate.py` | whether "94%" means anything |
| `apps/web/lib/signEngine.ts` | the same pipeline, in the browser |

`features.py` and `apps/web/lib/features.ts` must agree exactly. If they drift,
the browser feeds the model a distribution it has never seen, accuracy
collapses, and nothing reports an error.

---

## Full stack, locally

```bash
cp .env.example .env
docker compose up -d          # postgres, redis, livekit

cd apps/api && pip install -e . && uvicorn signspeak_api.main:app --reload
cd apps/web && npm install && npm run dev
```

Frontend `:3000`, API `:8000`, LiveKit `:7880`.

The web client refuses to start sign recognition without
`public/models/runtime_config.json`, which is produced by training. That
refusal is deliberate — a client that invents its own thresholds is a client
that speaks words it should have withheld.

---

## The ML pipeline

```bash
cd ml
pip install -e ".[capture,export]"

# 0. Test the plumbing before you own a camera
python scripts/make_synthetic.py --subjects 6 --per-class 24

# 1. Collect real data — run this on 6-8 different people, in different rooms
python -m signspeak_ml.capture --subject alice --lighting normal --distance medium
python -m signspeak_ml.capture --subject bob   --dominant left --lighting dim

# 2. Train. Default protocol is leave-one-subject-out.
python -m signspeak_ml.train --data data/raw --compare

# 3. Evaluate: calibration, risk-coverage, robustness slices, baseline
python -m signspeak_ml.evaluate --data data/raw --checkpoint artifacts/gru.pt --baseline

# 4. Export to the browser
python -m signspeak_ml.export_onnx --checkpoint artifacts/gru.pt \
       --out ../apps/web/public/models
```

`train.py` **refuses to run** leave-one-subject-out with fewer than three
subjects, and tells you why. A random split on sign data leaks near-duplicates
across train and test and inflates accuracy by 15–30 points; that error is the
most common way projects like this fail in front of an examiner.

---

## The things worth knowing before you build on this

**Segmentation, not classification, is the hard problem.** A live camera
produces an unbroken stream. Six well-separated signs are easy to classify;
knowing *when* a sign started and stopped is not. It is also ~60% of the
end-to-end latency budget.

**Softmax is not probability.** A network reading 0.94 is typically right about
80% of the time, worse on an unfamiliar signer. `calibrate.py` fits a
temperature on held-out data and cuts calibration error 3–5× for five lines of
code. The displayed number is post-calibration.

**A closed-set classifier must output one of its labels.** Someone pushing
their hair back gets a word. Three layers defend against it: a trained `NONE`
class, an energy-based OOD score, and a signing-space gate in the segmenter.

**Recognition runs on the client.** Camera frames and landmarks never leave the
device. Per-call AI cost is zero, capacity is browsers rather than GPUs, and
recognition keeps working when the network degrades — it reads the local track,
not the transmitted one. The speech direction is different: Chrome's Web Speech
API sends audio to Google. Say that asymmetry out loud rather than letting one
privacy claim cover both.

**The worst subject is the real accuracy.** Report mean, standard deviation and
minimum across folds. The minimum answers "does this work for the next person",
and it is usually 10–20 points below the mean.

---

## Roadmap, reordered

The original plan front-loads infrastructure and lands integration in the last
two weeks. Invert it — get the ugly complete loop working early, then make each
piece good.

1. **Week 1–2** — two browsers connected, one hardcoded sign spoken aloud, one
   caption rendered. Ugly, complete, end to end.
2. **Week 3–5** — collect the dataset. Six to eight people. This is the
   bottleneck and it is the only task that cannot be compressed later.
3. **Week 5–7** — train, calibrate, evaluate. Compare against the DTW baseline.
4. **Week 7–9** — segmentation on continuous video; measure it separately.
5. **Week 9–11** — accessibility pass, confidence UX, transcript ordering.
6. **Week 11–12** — human evaluation, robustness slices, write-up.

Auth is not on the critical path. Anonymous rooms work by design; forcing a
Deaf user to register before taking an urgent call is an accessibility failure
dressed up as security.

---

## Scope discipline

Things this repo deliberately does not claim:

- full or continuous sign-language translation
- support for Indian Sign Language (the architecture is adaptable; the data,
  evaluation and native-signer involvement do not exist yet)
- readiness for healthcare, legal or emergency use
- fully offline operation — *cloud-assisted initially, edge-capable by design*

Sign languages are not universal. ISL is a distinct language with its own
grammar, and any ISL claim needs ISL data and Deaf ISL signers involved in
building and evaluating it, not a vocabulary swap.
