/**
 * The sign recognition pipeline, client-side, end to end.
 *
 *   camera frame
 *     -> MediaPipe (hand + pose + face landmarkers)
 *     -> body-anchored features            features.ts
 *     -> streaming segmentation            this file, Segmenter
 *     -> ONNX temporal classifier          this file, Classifier
 *     -> temperature scaling + energy OOD  decision.ts
 *     -> accept / confirm / reject
 *
 * Everything here runs in the browser. No frame, no landmark and no feature
 * vector is ever sent to a server; only the decided word travels, over the
 * LiveKit data channel. That is what makes the privacy claim in the pitch
 * true rather than aspirational, and it is worth protecting - if you ever add
 * "just send the landmarks to the backend for debugging", the claim dies.
 */

import {
  FEATURE_DIM,
  PER_HAND,
  STATIC_DIM,
  addVelocity,
  encodeFrame,
  frameEnergy,
  handsInSigningSpace,
  resample,
  type RawFrame,
} from "./features";
import { type Decision, DecisionPolicy, type RuntimeConfig } from "./decision";

type SegState = "idle" | "maybeActive" | "active" | "maybeDone";

export interface Segment {
  frames: Float32Array[];
  startedAtMs: number;
  endedAtMs: number;
  peakEnergy: number;
  truncated: boolean;
}

/**
 * Motion-energy segmentation with hysteresis.
 *
 * Mirrors `ml/signspeak_ml/segment.py`. The single-threshold version of this
 * looks fine on a whiteboard and produces three detections per sign in
 * practice, because energy dithers across the line mid-sign. Enter high, exit
 * low, and require N consecutive frames on each side.
 *
 * The cost of the exit rule is pure latency: we cannot know a sign has ended
 * until we have seen exitFrames of stillness, which at 25fps is ~280ms added to
 * every single utterance. That is the largest term in the end-to-end budget -
 * larger than inference by two orders of magnitude - and it is the number to
 * attack if the demo feels sluggish.
 */
export class Segmenter {
  private state: SegState = "idle";
  private buffer: Float32Array[] = [];
  private stamps: number[] = [];
  private counter = 0;
  private prev: Float32Array | null = null;
  private activeStart: number | null = null;
  private peak = 0;

  constructor(private cfg: RuntimeConfig) {}

  reset() {
    this.state = "idle";
    this.buffer = [];
    this.stamps = [];
    this.counter = 0;
    this.prev = null;
    this.activeStart = null;
    this.peak = 0;
  }

  /** True while a sign is in progress - drives the "recognizing..." indicator.
   * Showing this matters: a system that silently decides whether you are
   * signing, without telling you, is unusable. */
  get isActive() {
    return this.state === "active" || this.state === "maybeDone";
  }

  push(frame: Float32Array | null, tMs: number): Segment | null {
    const s = this.cfg.segmenter;

    if (!frame) {
      // Torso lost. Treat as silence and close rather than stitching across a
      // gap of unknown length.
      const out = this.isActive ? this.close(tMs, false) : null;
      this.state = "idle";
      this.prev = null;
      this.counter = 0;
      this.activeStart = null;
      return out;
    }

    const e = frameEnergy(this.prev, frame);
    this.prev = frame;

    const inSpace = s.require_hands_in_signing_space ? handsInSigningSpace(frame) : true;
    const loud = e > s.enter_threshold && inSpace;
    const quiet = e < s.exit_threshold || !inSpace;

    this.buffer.push(frame);
    this.stamps.push(tMs);

    // Trimming shifts every index in the buffer, and `activeStart` is an index
    // INTO this buffer — so it must move with it. Miss this and segments break
    // about four seconds into a real call, while every short test still passes.
    const keep = this.cfg.window.max_segment_frames + s.pad_frames + 4;
    if (this.buffer.length > keep) {
      const drop = this.buffer.length - keep;
      this.buffer.splice(0, drop);
      this.stamps.splice(0, drop);
      if (this.activeStart !== null) this.activeStart = Math.max(0, this.activeStart - drop);
    }

    switch (this.state) {
      case "idle":
        if (loud) {
          this.state = "maybeActive";
          this.counter = 1;
          // Pre-roll: the start of a sign is low-energy by definition, so the
          // frames just before the threshold crossing are part of it.
          this.activeStart = Math.max(0, this.buffer.length - 1 - s.pad_frames);
          this.peak = e;
        }
        return null;

      case "maybeActive":
        if (loud) {
          this.counter++;
          this.peak = Math.max(this.peak, e);
          if (this.counter >= s.enter_frames) this.state = "active";
        } else {
          this.state = "idle";
          this.counter = 0;
          this.activeStart = null;
        }
        return null;

      case "active":
        this.peak = Math.max(this.peak, e);
        if (quiet) {
          this.state = "maybeDone";
          this.counter = 1;
        }
        return this.maybeTruncate(tMs);

      case "maybeDone":
        if (loud) {
          this.state = "active";
          this.counter = 0;
          this.peak = Math.max(this.peak, e);
          return this.maybeTruncate(tMs);
        }
        this.counter++;
        if (this.counter >= s.exit_frames) return this.close(tMs, false);
        return this.maybeTruncate(tMs);
    }
  }

  private maybeTruncate(tMs: number): Segment | null {
    if (this.activeStart === null) return null;
    if (this.buffer.length - this.activeStart >= this.cfg.window.max_segment_frames) {
      return this.close(tMs, true);
    }
    return null;
  }

  private close(tMs: number, truncated: boolean): Segment | null {
    const start = this.activeStart;
    this.state = "idle";
    this.counter = 0;
    this.activeStart = null;
    if (start === null) return null;

    const s = this.cfg.segmenter;
    const tail = truncated ? 0 : Math.max(0, s.exit_frames - s.pad_frames);
    const end = Math.max(start + 1, this.buffer.length - tail);
    const frames = this.buffer.slice(start, end);
    const startedAtMs = this.stamps[start];
    const endedAtMs = this.stamps[Math.min(end, this.stamps.length) - 1] ?? tMs;

    // Keep only a short pre-roll for the next sign, so the buffer does not
    // accumulate the whole call.
    const tailKeep = s.pad_frames + 2;
    if (this.buffer.length > tailKeep) {
      this.buffer.splice(0, this.buffer.length - tailKeep);
      this.stamps.splice(0, this.stamps.length - tailKeep);
    }

    if (frames.length < this.cfg.window.min_segment_frames) return null;

    const seg: Segment = {
      frames,
      startedAtMs,
      endedAtMs,
      peakEnergy: this.peak,
      truncated,
    };
    this.peak = 0;
    return seg;
  }
}

/** ONNX Runtime Web wrapper. WebGPU when available, WASM otherwise. */
export class Classifier {
  private session: any = null;
  private ort: any = null;

  constructor(private cfg: RuntimeConfig, private modelUrl: string) {}

  async load(): Promise<void> {
    this.ort = await import("onnxruntime-web");
    // WebGPU is ~3-5x faster here, but not everywhere, and a silent fallback
    // beats a blank screen on a machine without it.
    const providers = ["webgpu", "wasm"];
    try {
      this.session = await this.ort.InferenceSession.create(this.modelUrl, {
        executionProviders: providers,
        graphOptimizationLevel: "all",
      });
    } catch {
      this.session = await this.ort.InferenceSession.create(this.modelUrl, {
        executionProviders: ["wasm"],
      });
    }
  }

  get ready() {
    return this.session !== null;
  }

  /** Raw logits. Deliberately not softmax: the energy OOD score needs logits,
   * and temperature scaling has to happen before the softmax. */
  async logits(segmentFrames: Float32Array[]): Promise<{ logits: Float32Array; ms: number }> {
    if (!this.session) throw new Error("classifier not loaded");
    const T = segmentFrames.length;
    const withVel = addVelocity(segmentFrames);
    const x = resample(withVel, T, FEATURE_DIM, this.cfg.frames);

    const t0 = performance.now();
    const tensor = new this.ort.Tensor("float32", x, [1, this.cfg.frames, FEATURE_DIM]);
    const out = await this.session.run({ features: tensor });
    return { logits: out.logits.data as Float32Array, ms: performance.now() - t0 };
  }
}

export interface EngineEvent {
  decision: Decision;
  segmentFrames: number;
  startedAtMs: number;
  endedAtMs: number;
  inferenceMs: number;
  /** Sign end -> decision available. Excludes TTS and the network hop; add
   * those before quoting an end-to-end figure. */
  pipelineMs: number;
}

export interface EngineCallbacks {
  onEvent: (e: EngineEvent) => void;
  onActivity?: (active: boolean) => void;
  onTrackingChange?: (tracked: boolean) => void;
  onError?: (err: Error) => void;
}

/**
 * Drives the whole client-side pipeline off a video element.
 *
 * Runs detection on a timer rather than requestAnimationFrame: rAF is tied to
 * the display refresh (often 120Hz), which would run MediaPipe four times per
 * captured frame and heat the laptop for nothing. We pace at the camera's
 * actual rate.
 */
export class SignEngine {
  private segmenter: Segmenter;
  private classifier: Classifier;
  private policy: DecisionPolicy;
  private landmarker: any = null;
  private running = false;
  private lastEmit = new Map<string, number>();
  private timer: any = null;
  private t0 = 0;

  constructor(
    private cfg: RuntimeConfig,
    modelUrl: string,
    private dominant: "left" | "right",
    private cb: EngineCallbacks,
  ) {
    this.segmenter = new Segmenter(cfg);
    this.classifier = new Classifier(cfg, modelUrl);
    this.policy = new DecisionPolicy(cfg);
  }

  async load(landmarkerFactory: () => Promise<any>) {
    const [lm] = await Promise.all([landmarkerFactory(), this.classifier.load()]);
    this.landmarker = lm;
  }

  start(video: HTMLVideoElement) {
    if (this.running) return;
    this.running = true;
    this.t0 = performance.now();
    const interval = 1000 / this.cfg.window.target_fps;
    let lastActive = false;
    let lastTracked = true;
    let busy = false;

    this.timer = setInterval(async () => {
      if (!this.running || busy || video.readyState < 2) return;
      busy = true;
      try {
        const now = performance.now() - this.t0;
        const raw: RawFrame = this.landmarker.detect(video, now);
        const frame = encodeFrame(raw, this.dominant);

        const tracked = frame !== null;
        if (tracked !== lastTracked) {
          lastTracked = tracked;
          this.cb.onTrackingChange?.(tracked);
        }

        const seg = this.segmenter.push(frame, now);

        if (this.segmenter.isActive !== lastActive) {
          lastActive = this.segmenter.isActive;
          this.cb.onActivity?.(lastActive);
        }

        if (seg) await this.handleSegment(seg);
      } catch (err) {
        this.cb.onError?.(err as Error);
      } finally {
        busy = false;
      }
    }, interval);
  }

  stop() {
    this.running = false;
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
    this.segmenter.reset();
  }

  private async handleSegment(seg: Segment) {
    const tStart = performance.now();
    const { logits, ms } = await this.classifier.logits(seg.frames);
    const decision = this.policy.decide(logits);

    // Suppress an identical label repeating within the debounce window. Real
    // signers do repeat words, so this is a trade: too long and you swallow a
    // genuine "yes, yes", too short and one sign detected twice becomes a
    // stutter in the hearing user's ear.
    if (decision.action === "speak") {
      const last = this.lastEmit.get(decision.label) ?? -Infinity;
      if (seg.endedAtMs - last < this.cfg.decision.repeat_debounce_ms) return;
      this.lastEmit.set(decision.label, seg.endedAtMs);
    }

    this.cb.onEvent({
      decision,
      segmentFrames: seg.frames.length,
      startedAtMs: seg.startedAtMs,
      endedAtMs: seg.endedAtMs,
      inferenceMs: ms,
      pipelineMs: performance.now() - tStart,
    });
  }
}

export { STATIC_DIM, FEATURE_DIM, PER_HAND };
