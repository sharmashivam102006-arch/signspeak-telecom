/**
 * Confidence, abstention and the accept/confirm/reject policy.
 *
 * Mirrors `ml/signspeak_ml/calibrate.py`. This is the single place in the
 * client that decides whether a word gets spoken out loud. Keeping it in one
 * file, rather than scattered across components, is the difference between a
 * system you can reason about and one that occasionally announces "EMERGENCY"
 * in a bank branch for reasons nobody can reconstruct.
 *
 * Three guards, applied in order:
 *
 *   1. NONE class        the trained background class - movement you
 *                        anticipated and recorded
 *   2. energy OOD        logsumexp of the logits; catches movement you did not
 *                        anticipate, which is most of it
 *   3. calibrated conf   temperature-scaled softmax against thresholds fitted
 *                        to an explicit error budget
 *
 * The number shown on screen is post-temperature. A raw softmax reading 0.94
 * is typically right about 80% of the time, and worse on a signer the model
 * has never seen. Displaying that untouched tells the hearing participant
 * something false about how much to trust the word they just heard, in a
 * conversation where they have no other way to check.
 */

export interface Thresholds {
  accept: number;
  confirm: number;
  energy: number;
  temperature: number;
  achieved_risk?: number;
  achieved_coverage?: number;
}

export interface RuntimeConfig {
  version: number;
  model: string;
  onnx: string;
  vocab: string[];
  speechTemplates: Record<string, string>;
  frames: number;
  featureDim: number;
  staticDim: number;
  temperature: number;
  thresholds: Thresholds;
  segmenter: {
    enter_threshold: number;
    exit_threshold: number;
    enter_frames: number;
    exit_frames: number;
    pad_frames: number;
    require_hands_in_signing_space: boolean;
  };
  window: {
    target_fps: number;
    frames: number;
    min_segment_frames: number;
    max_segment_frames: number;
  };
  decision: {
    accept: number;
    confirm: number;
    consecutive_agreements: number;
    repeat_debounce_ms: number;
  };
}

export type Action = "speak" | "confirm" | "reject";

export interface Decision {
  action: Action;
  label: string;
  /** Word the hearing participant hears. A template lookup, not translation. */
  spoken: string;
  confidence: number;
  runnerUp: string | null;
  runnerUpConfidence: number;
  energy: number;
  reason: string;
  probs: number[];
}

export function softmax(logits: Float32Array | number[], temperature = 1): number[] {
  const t = Math.max(temperature, 1e-6);
  const scaled = Array.from(logits, (v) => v / t);
  const max = Math.max(...scaled);
  const exps = scaled.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

/**
 * Free energy: -T * logsumexp(logits / T). Lower means more in-distribution.
 *
 * Stronger than max-softmax for out-of-distribution detection and it needs
 * nothing at training time, which matters because you cannot enumerate
 * everything a person might do in front of a camera. Use it *alongside* the
 * NONE class: NONE catches what you thought to record, this catches the rest.
 */
export function energyScore(logits: Float32Array | number[], temperature = 1): number {
  const t = Math.max(temperature, 1e-6);
  const scaled = Array.from(logits, (v) => v / t);
  const max = Math.max(...scaled);
  const lse = max + Math.log(scaled.reduce((a, v) => a + Math.exp(v - max), 0));
  return -t * lse;
}

export class DecisionPolicy {
  private recent: string[] = [];

  constructor(private cfg: RuntimeConfig) {}

  reset() {
    this.recent = [];
  }

  decide(logits: Float32Array): Decision {
    const th = this.cfg.thresholds;
    const T = th.temperature ?? this.cfg.temperature ?? 1;
    const probs = softmax(logits, T);

    const order = probs
      .map((p, i) => [p, i] as [number, number])
      .sort((a, b) => b[0] - a[0]);
    const [conf, top] = order[0];
    const [runnerConf, runnerIdx] = order[1] ?? [0, -1];

    const label = this.cfg.vocab[top];
    const runnerUp = runnerIdx >= 0 ? this.cfg.vocab[runnerIdx] : null;
    const energy = energyScore(logits, T);
    const spoken = this.cfg.speechTemplates[label] ?? label;

    const base = {
      label,
      spoken,
      confidence: conf,
      runnerUp,
      runnerUpConfidence: runnerConf,
      energy,
      probs,
    };

    // 1. background class
    if (top === 0 || label === "NONE") {
      this.recent = [];
      return { ...base, action: "reject", reason: "background movement" };
    }

    // 2. out of distribution
    if (energy > th.energy) {
      this.recent = [];
      return {
        ...base,
        action: "reject",
        reason: "doesn't match any sign the model knows",
      };
    }

    // 3. calibrated confidence
    if (conf >= th.accept) {
      // Optional N-in-a-row agreement. Raising this trades latency for
      // accuracy: every extra repetition is another full sign the user has to
      // perform before anything is spoken. Default 1 - the calibrated
      // threshold is already doing the work.
      const need = this.cfg.decision.consecutive_agreements;
      if (need > 1) {
        this.recent.push(label);
        if (this.recent.length > need) this.recent.shift();
        if (this.recent.length < need || !this.recent.every((l) => l === label)) {
          return {
            ...base,
            action: "confirm",
            reason: `confident, waiting for ${need} in a row`,
          };
        }
        this.recent = [];
      }
      return { ...base, action: "speak", reason: "above accept threshold" };
    }

    if (conf >= th.confirm) {
      this.recent = [];
      return { ...base, action: "confirm", reason: "ambiguous" };
    }

    this.recent = [];
    return { ...base, action: "reject", reason: "below confirm threshold" };
  }
}

/** What the SIGNER sees. Never surface a guess as fact - if the system is not
 * sure, the person whose words are being spoken for them is the one who gets
 * to decide. */
export function signerMessage(d: Decision): string {
  switch (d.action) {
    case "speak":
      return d.label;
    case "confirm":
      return `Possible: ${d.label}. Sign again to confirm.`;
    case "reject":
      return d.reason === "background movement"
        ? ""
        : "Didn't catch that - please sign again.";
  }
}

/** What the HEARING participant sees next to the spoken word.
 *
 * They hear a confident voice say "I need help" with no way to judge it. A
 * visible confidence band is the only signal they get that the machine was
 * unsure, so it is not decoration - it is the thing that stops a
 * misrecognition being taken at face value.
 */
export function confidenceBand(c: number): { label: string; tone: "high" | "medium" | "low" } {
  if (c >= 0.9) return { label: "High confidence", tone: "high" };
  if (c >= 0.7) return { label: "Moderate confidence", tone: "medium" };
  return { label: "Low confidence", tone: "low" };
}

export async function loadRuntimeConfig(url = "/models/runtime_config.json"): Promise<RuntimeConfig> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(
      `No runtime_config.json at ${url}. Train a model and run ` +
        `\`python -m signspeak_ml.export_onnx\` - the client refuses to invent ` +
        `thresholds, because a guessed threshold is how a system ends up speaking ` +
        `words it should have withheld.`,
    );
  }
  return res.json();
}
