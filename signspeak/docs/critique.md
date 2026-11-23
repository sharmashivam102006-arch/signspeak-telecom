# A hard read of the SignSpeak plan

This is the document to argue with. The plan you wrote is well above the median
for a project at this stage — the scoping is honest, the refusal to claim full
sign-language translation is right, and the layered architecture is the correct
shape. What follows is what I think is wrong with it, ordered by how much
damage each one does if it goes unaddressed.

---

## 1. The ML task you chose is the easy part, and it will not carry the project

Six well-separated signs, recorded by you, classified from body-anchored
landmarks. In that setting a DTW k-nearest-neighbour classifier with no
training at all typically reaches 90–96% on a signer-dependent split. Your
GRU will also reach 90-something, and you will not be able to tell whether it
is better, because the gap between the two will be smaller than the variance
across folds.

That is not a reason to abandon the model. It is a reason to stop treating
"we built a temporal model that recognises signs" as the contribution, because
an examiner who knows the field will not treat it as one.

**What is actually hard, and absent from your plan:**

| Problem | Why it's hard | Where it lives here |
| --- | --- | --- |
| Segmentation | A live camera produces an unbroken stream. Nothing tells you when a sign starts or ends. | `ml/signspeak_ml/segment.py` |
| Open-set rejection | A closed-set classifier *must* emit one of your labels. A user scratching their nose gets announced as a word. | `calibrate.py`, energy score |
| Calibration | Softmax is not probability. Your "94%" is, untreated, roughly 80% accurate. | `calibrate.py`, temperature scaling |
| Signer independence | Random splits leak. Your real accuracy is the one on a person the model has never seen. | `dataset.py`, leave-one-subject-out |
| End-to-end latency | Dominated by segmentation lag, not inference — which inverts the usual optimisation instinct. | `docs/architecture.md` |

Reframe the contribution as: **a confidence-aware, signer-independent sign
recognition pipeline for live conversation, with a calibrated abstention
policy and a measured latency budget.** That is a defensible claim about work
that is genuinely non-trivial. "We built a sign classifier" is not.

---

## 2. The dataset split will silently inflate your numbers by 15–30 points

This is the single most likely way this project ends badly, and it happens in
about a third of student sign-recognition projects.

You record 30 takes of HELLO. You shuffle everything and take a random 25% as
the test set. Some of your training takes and some of your test takes are the
*same person performing the same sign minutes apart* — near-duplicates. The
model memorises your proportions, your timing, your habits. You report 97%.

Then the examiner signs HELLO and nothing happens.

**Fix, and it is not optional:** split by *person*. `leave_one_subject_out()`
in `dataset.py` is the default in `train.py` and it refuses to run with fewer
than three subjects. Report the signer-independent mean, the standard
deviation across folds, and — most importantly — the **worst subject's**
accuracy. That last number is the one that answers "does this work for the
next person who tries it", and it is usually 10–20 points below the mean.

Quote the random-split number too, explicitly labelled as signer-dependent,
*next to* the honest one. The gap between them is itself a finding worth a
paragraph.

---

## 3. "94% confidence" is currently a false statement

You gate speech output on a confidence score, which is the right instinct.
But a raw softmax from a neural network is systematically overconfident, and
more so on data unlike its training set — exactly the situation in deployment.

The consequence is specific and bad. A hearing person hears a fluent synthetic
voice say "I need help" and has no way to judge it. The number on screen is
their only signal about how much to trust it. If that number says 94 and the
system is right 80% of the time, you have built a machine that misleads people
about how much to trust a machine, in conversations where one party cannot
verify what was said.

**Fix:** temperature scaling. One scalar, fitted on a held-out split, five
lines of code, typically cuts expected calibration error by 3–5×. It cannot
change which class wins, so accuracy is untouched. Report ECE before and after
and include a reliability diagram. `calibrate.py` does all of this.

Then pick your thresholds from the **risk–coverage curve** against a stated
error budget ("at most 2% of spoken outputs may be wrong") instead of by
choosing 0.90 because it looks like a good number. `fit_thresholds()` does it.

---

## 4. You have no NONE class, and a closed-set classifier in a live call is dangerous

Your vocabulary is HELLO, YES, NO, HELP, THANK_YOU, EMERGENCY. Your classifier
outputs a distribution over six classes that sums to one. There is no option
for "that wasn't a sign".

So when a user pushes their hair back, the model returns a distribution over
six words. Sometimes it returns 0.91 on EMERGENCY. In a bank branch.

**Fix, three layers:**

1. A trained `NONE` class — record scratching, waving, reaching, gesturing
   while talking, and signs outside the vocabulary. This is in `config.VOCAB`
   at index 0 and it is not optional.
2. An energy-based OOD score (`energy_score()`), which catches movements you
   never thought to record. This matters because you cannot enumerate what a
   stranger does in front of a camera.
3. The signing-space gate in the segmenter: hands below the hip line never
   reach the classifier at all.

---

## 5. "Sign → AI understands → Speech" is not what your MVP does

Your pipeline diagram has a "Language Model" box between sign recognition and
text. In the MVP, that box is a Python dictionary:

```python
SPEECH_TEMPLATES = {"HELP": "I need help.", ...}
```

That is fine. Template lookup is the correct engineering choice for six
isolated signs. What is not fine is calling it understanding or translation in
a report, because a reviewer will ask one question — "what does the language
model do?" — and the honest answer will undercut everything around it.

Say instead: *isolated sign classification with templated speech output.*
Then, if you want the LLM box to be real, give it a real job: gloss-to-fluent-
sentence generation. But that requires *continuous* recognition producing a
gloss sequence, which requires CTC or a transducer, which is a different
project. Keep it in future work, where you already correctly put ISL
generation.

---

## 6. Latency: you will optimise the wrong thing

Rough budget at 25fps, measured on the pipeline in this repo:

```
MediaPipe landmarks (per frame, overlapped)          8 – 15 ms
end-of-sign detection (7 quiet frames @ 25fps)     ~ 280 ms     <-- dominates
classifier inference (small GRU, ONNX Web)           2 –  5 ms
TTS start (browser speechSynthesis)                 50 – 150 ms
WebRTC audio path                                  100 – 200 ms
                                                   -----------
sign finished -> hearing user hears it            ~ 450 – 650 ms
```

The classifier is under 1% of that. Spending three weeks making the model
faster changes nothing a user can perceive. The lever is the 280ms detection
lag, and shortening it starts cutting slow signs in half — a real trade-off
with a real curve you can plot.

**This is a good result and you should write it up as one.** "We found that
end-to-end latency in segment-then-classify sign recognition is dominated by
end-of-utterance detection, not inference, and characterised the
accuracy/latency trade-off" is a genuine, quotable finding. Drag the
end-of-sign slider in `demo/index.html` and watch the number move.

If you want to attack it properly, the answer is early-commit streaming
classification: classify the growing prefix every few frames and emit as soon
as confidence crosses threshold, without waiting for the sign to end. Note
that this forces the GRU to be **unidirectional** — the bidirectional accuracy
you measured offline stops being the accuracy users experience.

---

## 7. Two people, eight subsystems, one semester

Your split (Person 1 all AI, Person 2 all product) has a failure mode: LiveKit,
auth, Postgres, Redis, WebSocket ordering and accessible UI will consume
Person 2's entire semester, and the integration everything depends on lands in
the last two weeks.

Concretely, from this repo: **auth is not on the MVP critical path.** Anonymous
rooms work (`rooms.py` allows `user=None` deliberately), and forcing a Deaf
user to register before taking an urgent call is an accessibility failure
dressed up as security. Ship anonymous, add accounts when preferences need
persisting.

Suggested reorder: get the *boring* end-to-end loop working in week 3 — two
browsers, a hardcoded sign, a spoken word — before either of you builds
anything good. An ugly complete pipeline in week 3 beats two excellent halves
in week 12. You've noted yourself that you over-plan; this is the place it will
cost you.

---

## 8. Smaller things, still worth fixing

- **MediaPipe Holistic is deprecated.** Use the Tasks API landmarkers
  (`HandLandmarker` + `PoseLandmarker` + `FaceLandmarker`). Three graphs cost
  slightly more CPU and give noticeably better hand tracking, which is the
  input to everything.
- **Handedness.** ~10% of signers are left-dominant, and to a model fed raw
  left/right channels a left-dominant HELLO is a different sign. Canonicalise
  by mirroring (`canonicalize_handedness`), or collect double the data.
- **MediaPipe's handedness labels are image-relative.** The self-view is
  mirrored, so MediaPipe's "Left" is the user's *right* hand. Getting this
  backwards degrades the model subtly rather than obviously — you lose a week.
- **Transcript ordering.** Sign events arrive later than speech events by
  construction. Append in arrival order and your demo transcript reads
  backwards — and your demo script in §18 has exactly the alternating
  structure that triggers it. See `services/ordering.py`.
- **TTS plays on the wrong machine.** `speechSynthesis.speak()` uses the local
  speakers: audible to the signer (who can't hear it) and inaudible to the
  hearing participant (who needs it). Works in a demo with two laptops in one
  room; fails the moment the call is genuinely remote. Fix is
  `synthesizeToTrack()` — render TTS into a MediaStream and publish it as a
  separate LiveKit audio track.
- **Autoplay policy.** `speechSynthesis` is silently ignored until a real user
  gesture. Unlock it in the Join click handler or spend an afternoon debugging
  a classifier that was working fine.
- **Web Speech API sends audio to Google.** Your privacy story is true for the
  sign direction and false for the speech direction. State the asymmetry
  explicitly rather than letting one claim cover both.
- **`prefers-reduced-motion` and pinch-zoom.** Don't disable user scaling in
  the viewport meta tag. In an accessibility product that is an own goal.

---

## 9. What is genuinely good, and worth defending

Since the above is relentless — these are right and you should not let a
reviewer talk you out of them:

- **Client-side landmark extraction.** Raw video never leaving the device is a
  real architectural property, not marketing. It makes the healthcare and
  banking positioning defensible and it is cheap.
- **Refusing to claim full sign-language translation.** Most projects at this
  level overclaim here and get taken apart for it.
- **"Cloud-assisted initially, edge-capable by design."** Precisely the right
  register.
- **Confidence-aware UX as a first-class concern.** Your instinct that a wrong
  translation is worse than no translation is the correct product instinct, and
  it is what most of the engineering in this repo exists to serve.
- **Sign languages are not universal, ISL needs its own data.** Correct, and
  correctly deferred.
- **The product framing** — accessibility layer, not translator. Keep it.

---

## 10. The one-paragraph version

Your plan builds the right system and measures the wrong things. Keep the
architecture; change the evaluation. Split by person, calibrate the
confidence, add a background class, measure where the latency actually goes,
and report the worst subject rather than the mean. Do that, and a six-sign
vocabulary stops being a limitation you apologise for and becomes a controlled
setting in which you measured something real.
