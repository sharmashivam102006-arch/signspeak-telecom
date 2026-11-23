# Evaluation protocol

Phase 7 of your roadmap lists accuracy, precision, recall, F1, latency and
resource use. All necessary, none sufficient. This is what to measure instead,
and why each one earns its place.

---

## 0. Before any number exists: the dataset audit

Run `python -m signspeak_ml.train` and it prints `dataset_audit.json` first.
Put the **subject × class** table in the report. An overall sample count hides
imbalance and missing cells; the grid makes both obvious.

Target, minimum viable:

| dimension | minimum | comfortable |
| --- | --- | --- |
| subjects | 6 | 8–10 |
| takes per sign per subject | 15 | 25–30 |
| recording sessions per subject | 2 | 3 (different days, different rooms) |
| lighting conditions | 2 | 3–4 (normal, dim, backlit) |
| left-dominant signers | 1 | 2 |
| NONE samples | 1.5× a sign class | 2× |

Two sessions per subject matters more than it looks. One session captures one
outfit, one room, one position; the model learns those and you cannot tell,
because your test split has them too.

---

## 1. The headline: signer-independent accuracy

**Protocol:** leave-one-subject-out. `train.py` refuses fewer than three
subjects.

Report four numbers, not one:

- **mean** across folds
- **standard deviation** across folds
- **minimum** — the worst subject
- **macro F1** excluding NONE

The minimum is the number that answers "does this work for the next person who
tries it", and it is routinely 10–20 points below the mean. If the standard
deviation is larger than the gap between your models, say so: the model choice
is noise and the dataset is the bottleneck. That is a finding, not a failure.

Report the signer-*dependent* (random-split) number too, explicitly labelled,
next to it. The gap between the two quantifies the leak, and showing you
understand it is worth more than the inflated figure.

---

## 2. Calibration

Accuracy says how often it's right. Calibration says whether the confidence
number means anything — and your UI gates speech output on that number.

| metric | what it tells you | target |
| --- | --- | --- |
| ECE (15 bins) | average gap between stated and actual confidence | < 0.05 after scaling |
| MCE | worst bin's gap | < 0.15 |
| Brier score | overall probabilistic quality | lower is better |
| reliability diagram | *where* it's miscalibrated | should hug the diagonal |

Report ECE **before and after** temperature scaling. Typical small GRU on a
signer-independent split: 0.10–0.20 before, 0.02–0.05 after. That improvement
costs one scalar and five lines and is one of the easiest real results in the
project.

Fit the temperature on a **subject-disjoint slice of the training set**, never
on test (leakage) and never on data the model trained on (a trained model is
overconfident on its own training set by construction, so the temperature you'd
learn there is meaningless). `train.py` carves this out automatically.

---

## 3. Risk–coverage: the metric your product actually needs

The product question is not "how accurate is it" but **"if I only speak the
ones I'm most sure about, how often am I wrong, and how much do I refuse to
say?"** Accuracy cannot answer that. The risk–coverage curve can.

Sort predictions by calibrated confidence, keep the top fraction, plot error
rate against coverage. Then pick thresholds from a stated error budget:

```
max_risk = 0.02      # at most 2% of spoken outputs may be the wrong word
confirm_risk = 0.15  # looser bar for "Possible: HELLO — repeat to confirm"
```

`fit_thresholds()` returns the achieved risk and coverage alongside the
thresholds. Report both: "at a 2% error budget the system speaks 74% of
attempted signs" is a far more informative sentence than "accuracy 93%".

Both budgets are product decisions, not statistical ones. A classroom and a
bank branch want different numbers — which is the argument for exposing them as
settings rather than baking in 0.90.

---

## 4. Open-set behaviour

The failure that will actually embarrass you in a live demo.

**Protocol:** record a held-out set of non-vocabulary movement — scratching,
waving, adjusting glasses, reaching off-screen, gesturing while talking, and
3–5 real signs *outside* your vocabulary. None of it appears in training.

Measure:

- **False-acceptance rate**: fraction of OOD movements that produce a *spoken*
  output. Target < 5%. This is the number that decides whether the system is
  safe to switch on.
- **AUROC** of the energy score separating in-vocabulary from OOD.
- **Ablation**: NONE class only / energy score only / both. Show each layer
  earns its place.

---

## 5. Segmentation, evaluated on its own

Classification accuracy on pre-cut clips says nothing about performance on a
live stream. Segmentation is the difference, and almost nobody measures it.

**Protocol:** record continuous 2–3 minute sessions where a subject signs
naturally with pauses, interleaved with non-signing movement. Hand-label the
true spans. Run `segment_offline()`.

- **Detection F1 at IoU ≥ 0.5** (`segmentation_f1()`)
- **Boundary error**: mean |predicted start − true start| in ms
- **Over-segmentation rate**: true signs split into 2+ detections
- **Merge rate**: 2+ true signs merged into one detection (this is
  co-articulation, and it is where segment-then-classify structurally breaks)

Then report **end-to-end accuracy on continuous video** — the composition of
segmentation and classification. It will be well below your clip accuracy. That
gap is one of the most honest and interesting numbers you can produce.

---

## 6. Latency, decomposed

Do not report a single end-to-end figure. Decompose it, because the
decomposition is the result:

| stage | how to measure |
| --- | --- |
| landmark extraction | `performance.now()` around `detectForVideo` |
| end-of-sign detection | `exit_frames / fps` — deterministic |
| inference | `measure_latency()` in `evaluate.py`, p50/p95/p99 |
| TTS start | return value of `speak()` |
| network | LiveKit stats API |

Report p50 and p95, never the mean alone — a 2-second p99 is what users
remember.

Then plot **accuracy vs `exit_frames`**. This is the trade-off curve at the
heart of the system: patient segmentation is accurate and feels sluggish,
impatient segmentation is snappy and chops slow signs in half. Producing that
curve is a contribution in itself.

---

## 7. Robustness slices

The headline number hides the failure mode. `slice_metrics()` breaks accuracy
down by lighting, distance, background, dominant hand and subject.

A system at 92% overall but 61% in dim light is not a 92% system for anyone who
takes a call in the evening. Say that sentence in the report.

Slices to report: lighting, distance, background clutter, left vs right
dominant, per-subject, and — if you have it — with vs without face landmarks.

---

## 8. Baseline comparison

Run `evaluate.py --baseline`. If your GRU does not clearly beat DTW k-NN on the
**signer-independent** split, say so. "We compared three temporal architectures
on identical features and splits, and at this data scale the untrained
nearest-neighbour baseline was within one standard deviation of the best
neural model" is a real, publishable-shaped finding. Claiming a winner that the
error bars do not support is the thing that gets picked apart in a viva.

Report parameters and inference latency alongside accuracy. A model that is
0.4 points better and 40× larger has not won.

---

## 9. In-the-wild telemetry

`GET /calls/{id}/events/summary` gives what no offline test set can:

- **correction rate** — how often a human had to fix the system. Offline
  accuracy cannot measure this, because offline there is no human. This is
  arguably the most honest single number in the whole project.
- distribution of speak / confirm / reject
- real end-to-end latency p50 and p95
- the actual confusion pairs, from real corrections

Every correction is a labelled error from the real distribution — the kind of
data careful dataset collection cannot produce, because you cannot anticipate
what a stranger's living room looks like at 9pm.

---

## 10. Human evaluation (small, and worth more than it costs)

Even 4–6 participants. Half signers, half hearing, paired.

- **Task success**: can the pair complete a scripted exchange (greeting,
  request, thanks)?
- **Turns to success**: how many repetitions were needed?
- **Perceived trust**: does the hearing participant say they trusted the
  confidence display? Did a low-confidence marker change their behaviour?
- **SUS** or a short Likert set, administered identically to both roles.

If you can recruit even two Deaf or hard-of-hearing participants, their
feedback outweighs every number above. A system that scores 95% and that no
Deaf person wants to use has failed at the only thing it was for.

---

## Report table skeleton

```
Table 1  Dataset composition (subject × class × condition)
Table 2  Signer-independent accuracy: mean / std / min / macro-F1, per model
Table 3  Calibration: ECE, MCE, Brier, before and after temperature scaling
Table 4  Risk–coverage at 1%, 2%, 5%, 10% error budgets
Table 5  Open-set: false-acceptance rate, AUROC, ablation
Table 6  Segmentation: F1@IoU0.5, boundary error, over/under-segmentation
Table 7  Latency decomposition, p50 / p95 / p99
Table 8  Robustness slices
Table 9  Baseline comparison: accuracy, params, latency
Table 10 In-the-wild: correction rate, action distribution

Fig 1  Reliability diagram, before and after calibration
Fig 2  Risk–coverage curve with chosen operating point marked
Fig 3  Accuracy vs exit_frames (the latency/accuracy trade-off)
Fig 4  Confusion matrix, signer-independent
Fig 5  Per-subject accuracy, sorted — the spread is the story
```
