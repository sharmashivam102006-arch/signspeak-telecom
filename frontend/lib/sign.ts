// MediaPipe hand-landmark extraction + sequence buffer + backend prediction.
//
// VERIFIED 2026-11-17 in a real headless Chromium session:
//  * the legacy @mediapipe/hands `hands.js` CDN does NOT load in this
//    environment — the browser cannot reach jsdelivr at all (fetch fails),
//    even though curl from the shell can. Shipping it as the only loader
//    silently yields nothing.
//  * @mediapipe/tasks-vision HandLandmarker DOES work when its wasm +
//    model are bundled locally: the npm package is installed, its wasm is
//    copied to public/mediapipe/wasm/ and hand_landmarker.task to
//    public/mediapipe/ — same-origin, no CDN needed. That is the primary
//    loader. The legacy hands.js CDN is still attempted as a last resort
//    (it works on networks where the browser can reach jsdelivr), and if
//    both fail the caller is told plainly — nothing is faked.
//
// Sequence buffer stores RAW 21x3 landmarks per frame. Normalisation is
// server-side (the backend normalises before both the demo heuristics and
// the trained LSTM), so the frontend must NOT flatten to 63 floats — the
// backend rejects a flattened frame as no-hand (it needs 21 points).
//
// Predictions come ONLY from POST /api/ai/sign/predict and carry the
// backend's mode. The live path surfaces a prediction ONLY when
// mode === "trained-model". A 503 (model-not-trained) or a demo-mode
// response on a plain call means "not trained" — never a prediction.
// Demo heuristics run ONLY when the caller passes demo: true, and their
// results are always labelled DEMO.
export type Landmark = { x: number; y: number; z: number };

const HANDS_CDN = "https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js";
const CAMERA_CDN = "https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js";
const LOCAL_WASM_PATH = "/mediapipe/wasm";
const LOCAL_MODEL_PATH = "/mediapipe/hand_landmarker.task";

export type HandTrackerKind = "tasks-vision" | "hands-js";

export type HandTracker = {
  kind: HandTrackerKind;
  /** Start the frame loop feeding `video` frames to `onLandmarks`
   *  (called with the 21 landmarks, or null when no hand is visible). */
  start: (video: HTMLVideoElement, onLandmarks: (lm: Landmark[] | null) => void) => void;
  stop: () => void;
};

export type LoaderResult = { tracker: HandTracker | null; loader: HandTrackerKind | null; error: string };

declare global {
  interface Window {
    __signspeak_mediapipe?: { loader: HandTrackerKind | null; error: string };
  }
}

function loadScript(src: string): Promise<void> {
  return new Promise((resolve, reject) => {
    if (document.querySelector(`script[src="${src}"]`)) return resolve();
    const s = document.createElement("script");
    s.src = src; s.async = true;
    s.onload = () => resolve();
    s.onerror = () => reject(new Error(`Failed to load ${src}`));
    document.head.appendChild(s);
  });
}

type TasksVisionLike = {
  FilesetResolver: { forVisionTasks: (path: string) => Promise<unknown> };
  HandLandmarker: {
    createFromOptions: (fileset: unknown, opts: Record<string, unknown>) => Promise<{
      detectForVideo: (video: HTMLVideoElement, ts: number) => { landmarks?: { x: number; y: number; z: number }[][] };
      close: () => void;
    }>;
  };
};

async function createTasksVisionTracker(): Promise<HandTracker> {
  const vision = (await import("@mediapipe/tasks-vision")) as unknown as TasksVisionLike;
  const fileset = await vision.FilesetResolver.forVisionTasks(LOCAL_WASM_PATH);
  const landmarker = await vision.HandLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: LOCAL_MODEL_PATH, delegate: "CPU" },
    runningMode: "VIDEO",
    numHands: 1,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  let raf = 0;
  let stopped = false;
  return {
    kind: "tasks-vision",
    start(video, onLandmarks) {
      stopped = false;
      const loop = () => {
        if (stopped) return;
        raf = requestAnimationFrame(loop);
        if (video.readyState < 2) return;
        try {
          const res = landmarker.detectForVideo(video, performance.now());
          const lm = res.landmarks?.[0];
          onLandmarks(lm ? lm.slice(0, 21).map((p) => ({ x: p.x, y: p.y, z: p.z })) : null);
        } catch {
          // One bad frame must not kill the loop.
        }
      };
      loop();
    },
    stop() {
      stopped = true;
      cancelAnimationFrame(raf);
      try { landmarker.close(); } catch { /* ignore */ }
    },
  };
}

type HandsLike = {
  setOptions: (o: Record<string, unknown>) => void;
  onResults: (cb: (r: { multiHandLandmarks?: Landmark[][] }) => void) => void;
  send: (input: { image: HTMLVideoElement }) => Promise<void>;
  close: () => void;
};

async function createHandsJsTracker(): Promise<HandTracker> {
  await loadScript(HANDS_CDN);
  await loadScript(CAMERA_CDN);
  const Ctor = (window as unknown as { Hands?: new (o: { locateFile: (f: string) => string }) => HandsLike }).Hands;
  if (!Ctor) throw new Error("hands.js loaded but window.Hands is missing");
  const hands = new Ctor({ locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${f}` });
  hands.setOptions({ maxNumHands: 1, modelComplexity: 0, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
  let raf = 0;
  let stopped = false;
  let cb: ((lm: Landmark[] | null) => void) | null = null;
  hands.onResults((res) => {
    const lm = res.multiHandLandmarks?.[0];
    cb?.(lm ? lm.slice(0, 21) : null);
  });
  return {
    kind: "hands-js",
    start(video, onLandmarks) {
      cb = onLandmarks;
      stopped = false;
      const loop = async () => {
        if (stopped) return;
        try { await hands.send({ image: video }); } catch { /* frame skipped */ }
        if (!stopped) raf = requestAnimationFrame(loop);
      };
      loop();
    },
    stop() {
      stopped = true;
      cancelAnimationFrame(raf);
      try { hands.close(); } catch { /* ignore */ }
    },
  };
}

/** Load whichever hand tracker genuinely works in this browser.
 *  tasks-vision (local bundle) first, legacy hands.js CDN second.
 *  Returns tracker null + a plain-language error when neither works. */
export async function createHandTracker(): Promise<LoaderResult> {
  const errors: string[] = [];
  try {
    const tracker = await createTasksVisionTracker();
    window.__signspeak_mediapipe = { loader: "tasks-vision", error: "" };
    return { tracker, loader: "tasks-vision", error: "" };
  } catch (e) {
    errors.push(`HandLandmarker (local bundle): ${e instanceof Error ? e.message : String(e)}`);
  }
  try {
    const tracker = await createHandsJsTracker();
    window.__signspeak_mediapipe = { loader: "hands-js", error: "" };
    return { tracker, loader: "hands-js", error: "" };
  } catch (e) {
    errors.push(`MediaPipe Hands (CDN): ${e instanceof Error ? e.message : String(e)}`);
  }
  const error = `Hand tracking is unavailable in this browser. ${errors.join(" ")}`;
  window.__signspeak_mediapipe = { loader: null, error };
  return { tracker: null, loader: null, error };
}

/** Normalise 21 hand landmarks: wrist-relative, scale by wrist→middle-MCP distance.
 *  Kept for local use only — the prediction path sends raw landmarks and the
 *  backend normalises. Do not feed this flattened output to /predict. */
export function normaliseLandmarks(lm: Landmark[]): number[] {
  if (lm.length < 21) return [];
  const wrist = lm[0];
  const scale = Math.hypot(lm[9].x - wrist.x, lm[9].y - wrist.y) || 1;
  const out: number[] = [];
  for (const p of lm.slice(0, 21)) {
    out.push((p.x - wrist.x) / scale, (p.y - wrist.y) / scale, (p.z - wrist.z) / scale);
  }
  return out;
}

/** Did the hand actually move across this sequence? Guards the live path
 *  against re-predicting a held-still hand every throttle window. */
export function sequenceMoved(seq: Landmark[][], minMeanDisplacement = 0.02): boolean {
  if (seq.length < 2) return false;
  const first = seq[0];
  const last = seq[seq.length - 1];
  let sum = 0;
  for (let i = 0; i < 21; i++) {
    sum += Math.hypot(last[i].x - first[i].x, last[i].y - first[i].y);
  }
  return sum / 21 > minMeanDisplacement;
}

export class SequenceBuffer {
  private frames: Landmark[][] = [];
  constructor(private maxFrames = 30) {}
  /** Push one frame of RAW 21 landmarks (not flattened/normalised). */
  push(frame: Landmark[]) {
    if (!frame || frame.length < 21) return;
    this.frames.push(frame.slice(0, 21));
    if (this.frames.length > this.maxFrames) this.frames.shift();
  }
  get length() { return this.frames.length; }
  get ready() { return this.frames.length >= this.maxFrames; }
  snapshot(): Landmark[][] { return this.frames.map(f => [...f]); }
  lastFrame(): Landmark[] | null { return this.frames.length ? [...this.frames[this.frames.length - 1]] : null; }
  clear() { this.frames = []; }
}

// Smoothing / duplicate suppression shared by live + demo predictions.
export class PredictionSmoother {
  private lastSign = ""; private lastAt = 0; private streak = 0;
  /** Returns the sign to surface, or null if it should be suppressed as a duplicate. */
  accept(sign: string, confidence: number, minConfidence: number): { sign: string; confidence: number } | null {
    if (confidence < minConfidence) return null;
    const now = Date.now();
    if (sign === this.lastSign && now - this.lastAt < 5000) return null;
    if (sign === this.lastSign) this.streak += 1; else this.streak = 1;
    this.lastSign = sign; this.lastAt = now;
    return { sign, confidence };
  }
  reset() { this.lastSign = ""; this.streak = 0; }
}

export type PredictOutcome =
  | { kind: "trained"; prediction: string; confidence: number; requiresConfirm: boolean; message: string; vocabulary: string[] }
  | { kind: "demo"; prediction: string; confidence: number; requiresConfirm: boolean; message: string; vocabulary: string[] }
  | { kind: "no-hand"; message: string }
  | { kind: "not-trained"; message: string }
  | { kind: "error"; message: string };

type BackendPredictResponse = {
  prediction?: string | null;
  confidence?: number;
  mode?: string;
  requires_confirm?: boolean;
  message?: string;
  vocabulary?: string[];
  detail?: string;
};

/** POST /api/ai/sign/predict with a landmark sequence.
 *  - 503 / mode "model-not-trained"  → { kind: "not-trained" }: no prediction is shown.
 *  - mode "trained-model"            → { kind: "trained" }: a real model prediction.
 *  - mode "demo-rule-based"          → { kind: "demo" } ONLY when opts.demo is true;
 *    on a plain live call it is treated as "not-trained" — the live path
 *    never surfaces demo heuristics as recognition.
 *  - no prediction / mode "no-hand"  → { kind: "no-hand" }.
 *  Never throws for a backend answer; only network-level failures become "error". */
export async function predictSign(
  sequence: Landmark[][],
  apiUrl: string,
  token: string | null,
  opts: { demo?: boolean; force?: boolean } = {},
): Promise<PredictOutcome> {
  const frames = sequence.map(frame => frame.slice(0, 21).map(p => [p.x, p.y, p.z]));
  let res: Response;
  try {
    res = await fetch(`${apiUrl}/api/ai/sign/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...(token ? { Authorization: `Bearer ${token}` } : {}) },
      body: JSON.stringify({ sequence: frames, demo: opts.demo ?? false, force: opts.force ?? false }),
    });
  } catch (e) {
    return { kind: "error", message: `Could not reach the sign service: ${e instanceof Error ? e.message : String(e)}` };
  }
  let data: BackendPredictResponse = {};
  try { data = (await res.json()) as BackendPredictResponse; } catch { /* non-JSON body */ }
  if (res.status === 503 || data.mode === "model-not-trained") {
    return { kind: "not-trained", message: data.message || data.detail || "Sign model is not trained yet." };
  }
  if (!res.ok) {
    return { kind: "error", message: data.detail || data.message || `Sign prediction failed (${res.status})` };
  }
  if (!data.prediction || data.mode === "no-hand") {
    return { kind: "no-hand", message: data.message || "No hand detected." };
  }
  const confidence = data.confidence ?? 0;
  if (data.mode === "trained-model") {
    return {
      kind: "trained",
      prediction: data.prediction,
      confidence,
      requiresConfirm: data.requires_confirm ?? confidence < 0.85,
      message: data.message ?? "",
      vocabulary: data.vocabulary ?? [],
    };
  }
  if (data.mode === "demo-rule-based") {
    if (opts.demo) {
      return {
        kind: "demo",
        prediction: data.prediction,
        confidence,
        requiresConfirm: data.requires_confirm ?? true,
        message: data.message ?? "",
        vocabulary: data.vocabulary ?? [],
      };
    }
    // Demo heuristics on a plain live call (old backend behaviour):
    // not a real prediction — report the model as not trained.
    return { kind: "not-trained", message: "Sign model is not trained yet — the backend only offers demo heuristics." };
  }
  return { kind: "error", message: data.message || `Unexpected prediction mode: ${data.mode ?? "unknown"}` };
}
