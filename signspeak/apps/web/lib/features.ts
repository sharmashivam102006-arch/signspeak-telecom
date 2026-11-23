/**
 * Landmark -> feature vector. Line-for-line mirror of
 * `ml/signspeak_ml/features.py`.
 *
 * These two files MUST agree. If the browser normalizes landmarks even
 * slightly differently from the trainer, the model receives inputs from a
 * distribution it has never seen, accuracy collapses, and nothing in the stack
 * reports an error - you just get a model that is mysteriously worse in
 * production than in the notebook. That failure is silent, common, and
 * expensive to debug.
 *
 * `ml/scripts/check_parity.py` asserts both implementations produce identical
 * vectors for the same input. Run it in CI. If you change one file, change
 * both in the same commit.
 */

export const NUM_HAND_LANDMARKS = 21;

export const POSE = {
  nose: 0,
  leftEye: 2,
  rightEye: 5,
  leftShoulder: 11,
  rightShoulder: 12,
  leftElbow: 13,
  rightElbow: 14,
  leftWrist: 15,
  rightWrist: 16,
  leftHip: 23,
  rightHip: 24,
} as const;

export const POSE_KEEP_ORDER = [0, 2, 5, 11, 12, 13, 14, 15, 16, 23, 24];

const WRIST = 0;
const INDEX_MCP = 5;
const MIDDLE_MCP = 9;
const PINKY_MCP = 17;

export const PER_HAND_SHAPE = NUM_HAND_LANDMARKS * 3; // 63
export const PER_HAND_POSITION = 3;
export const PER_HAND_ORIENTATION = 6;
export const PER_HAND_PRESENCE = 1;
export const PER_HAND =
  PER_HAND_SHAPE + PER_HAND_POSITION + PER_HAND_ORIENTATION + PER_HAND_PRESENCE; // 73
export const POSE_BLOCK = POSE_KEEP_ORDER.length * 3; // 33
export const FACE_BLOCK = 5;
export const STATIC_DIM = 2 * PER_HAND + POSE_BLOCK + FACE_BLOCK; // 184
export const FEATURE_DIM = STATIC_DIM * 2; // 368

const EPS = 1e-6;

const FACE_POINTS = {
  browLeft: 105,
  browRight: 334,
  eyeLeftTop: 159,
  eyeLeftBottom: 145,
  mouthTop: 13,
  mouthBottom: 14,
  mouthLeft: 61,
  mouthRight: 291,
  noseTip: 1,
  chin: 152,
} as const;

export type Vec3 = { x: number; y: number; z: number };
export type Landmarks = Vec3[];

export interface RawFrame {
  pose: Landmarks | null;
  leftHand: Landmarks | null;
  rightHand: Landmarks | null;
  face: Landmarks | null;
}

export interface BodyFrame {
  origin: [number, number, number];
  /** Body axes as rows: xAxis, yAxis, zAxis. */
  R: [number, number, number][];
  scale: number;
}

/* ------------------------------------------------------------------ */
/* vector helpers                                                      */
/* ------------------------------------------------------------------ */
const sub = (a: Vec3, b: Vec3): [number, number, number] => [a.x - b.x, a.y - b.y, a.z - b.z];
const norm = (v: number[]) => Math.hypot(v[0], v[1], v[2]);
const dot = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: number[], b: number[]): [number, number, number] => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const scale = (v: number[], s: number): [number, number, number] => [v[0] * s, v[1] * s, v[2] * s];

/**
 * Orthonormal frame anchored on the torso.
 *
 * Anchoring here rather than on the image removes camera distance, framing and
 * roll in one step - the three things that otherwise make the same sign look
 * numerically different every time the signer shifts in their chair.
 */
export function bodyFrame(pose: Landmarks | null): BodyFrame | null {
  if (!pose || pose.length < 25) return null;

  const ls = pose[POSE.leftShoulder];
  const rs = pose[POSE.rightShoulder];
  const lh = pose[POSE.leftHip];
  const rh = pose[POSE.rightHip];
  if (!ls || !rs || !lh || !rh) return null;

  const origin: [number, number, number] = [
    (ls.x + rs.x) / 2,
    (ls.y + rs.y) / 2,
    (ls.z + rs.z) / 2,
  ];
  const hipMid: [number, number, number] = [
    (lh.x + rh.x) / 2,
    (lh.y + rh.y) / 2,
    (lh.z + rh.z) / 2,
  ];

  let xAxis = sub(ls, rs);
  const s = norm(xAxis);
  if (s < EPS) return null;
  xAxis = scale(xAxis, 1 / s);

  const yRaw: [number, number, number] = [
    hipMid[0] - origin[0],
    hipMid[1] - origin[1],
    hipMid[2] - origin[2],
  ];
  // Gram-Schmidt against x so the frame is genuinely orthonormal.
  let yAxis: [number, number, number] = [
    yRaw[0] - dot(yRaw, xAxis) * xAxis[0],
    yRaw[1] - dot(yRaw, xAxis) * xAxis[1],
    yRaw[2] - dot(yRaw, xAxis) * xAxis[2],
  ];
  const ny = norm(yAxis);
  if (ny < EPS) return null;
  yAxis = scale(yAxis, 1 / ny);

  return { origin, R: [xAxis, yAxis, cross(xAxis, yAxis)], scale: s };
}

function toBody(p: Vec3, bf: BodyFrame): [number, number, number] {
  const d = [p.x - bf.origin[0], p.y - bf.origin[1], p.z - bf.origin[2]];
  return [dot(bf.R[0], d) / bf.scale, dot(bf.R[1], d) / bf.scale, dot(bf.R[2], d) / bf.scale];
}

/* ------------------------------------------------------------------ */
/* per-hand                                                            */
/* ------------------------------------------------------------------ */
function encodeHand(hand: Landmarks | null, bf: BodyFrame, out: Float32Array, offset: number) {
  if (!hand || hand.length !== NUM_HAND_LANDMARKS) {
    // Zeros AND presence = 0. Without the flag, "no hand" and "hand at the
    // body origin" are the same vector and the model learns nonsense from
    // tracking dropouts.
    out.fill(0, offset, offset + PER_HAND);
    return;
  }

  const hb = hand.map((p) => toBody(p, bf));
  const wrist = hb[WRIST];

  // handshape: wrist-relative, divided by hand size, so an adult and a child
  // making the same shape land in the same place
  const rel = hb.map((p) => [p[0] - wrist[0], p[1] - wrist[1], p[2] - wrist[2]]);
  let size = norm(rel[MIDDLE_MCP]);
  if (size < EPS) size = 1;
  let o = offset;
  for (const r of rel) {
    out[o++] = r[0] / size;
    out[o++] = r[1] / size;
    out[o++] = r[2] / size;
  }

  // location
  out[o++] = wrist[0];
  out[o++] = wrist[1];
  out[o++] = wrist[2];

  // orientation: palm normal + finger axis. Flip the palm and it is a
  // different sign, so this is signal, not decoration.
  const v1 = [rel[INDEX_MCP][0], rel[INDEX_MCP][1], rel[INDEX_MCP][2]];
  const v2 = [rel[PINKY_MCP][0], rel[PINKY_MCP][1], rel[PINKY_MCP][2]];
  let n = cross(v1, v2);
  const nn = norm(n);
  n = nn > EPS ? scale(n, 1 / nn) : [0, 0, 0];
  let f = [rel[MIDDLE_MCP][0], rel[MIDDLE_MCP][1], rel[MIDDLE_MCP][2]];
  const fn = norm(f);
  f = fn > EPS ? scale(f, 1 / fn) : [0, 0, 0];
  out[o++] = n[0];
  out[o++] = n[1];
  out[o++] = n[2];
  out[o++] = f[0];
  out[o++] = f[1];
  out[o++] = f[2];

  out[o] = 1; // present
}

/** Five scalars, not 478 points. Enough for non-manual markers, and it means
 * the system carries facial expression rather than a face. */
function encodeFace(face: Landmarks | null, bf: BodyFrame, out: Float32Array, offset: number) {
  if (!face || face.length <= FACE_POINTS.mouthRight) {
    out.fill(0, offset, offset + FACE_BLOCK);
    return;
  }
  const inv = 1 / Math.max(bf.scale, EPS);
  const p = (i: number) => face[i];

  const eyeMidY = (p(FACE_POINTS.eyeLeftTop).y + p(FACE_POINTS.eyeLeftBottom).y) / 2;
  const browY = (p(FACE_POINTS.browLeft).y + p(FACE_POINTS.browRight).y) / 2;
  const mouthMidX = (p(FACE_POINTS.mouthLeft).x + p(FACE_POINTS.mouthRight).x) / 2;

  out[offset] = (eyeMidY - browY) * inv;
  out[offset + 1] = Math.abs(p(FACE_POINTS.eyeLeftTop).y - p(FACE_POINTS.eyeLeftBottom).y) * inv;
  out[offset + 2] = Math.abs(p(FACE_POINTS.mouthTop).y - p(FACE_POINTS.mouthBottom).y) * inv;
  out[offset + 3] = Math.abs(p(FACE_POINTS.mouthLeft).x - p(FACE_POINTS.mouthRight).x) * inv;
  out[offset + 4] = (p(FACE_POINTS.noseTip).x - mouthMidX) * inv;
}

/** Negate body-frame x on every 3-vector: maps a left-dominant signer into the
 * right-dominant convention the model was trained on. */
export function mirrorStatic(v: Float32Array): Float32Array {
  const out = v.slice();
  let o = 0;
  for (let h = 0; h < 2; h++) {
    for (let i = 0; i < NUM_HAND_LANDMARKS; i++) out[o + i * 3] *= -1;
    o += PER_HAND_SHAPE;
    out[o] *= -1;
    o += PER_HAND_POSITION;
    out[o] *= -1;
    out[o + 3] *= -1;
    o += PER_HAND_ORIENTATION + PER_HAND_PRESENCE;
  }
  for (let i = 0; i < POSE_KEEP_ORDER.length; i++) out[o + i * 3] *= -1;
  o += POSE_BLOCK;
  out[o + 4] *= -1; // head yaw
  return out;
}

/**
 * One frame -> STATIC_DIM floats, or null when the torso is not visible.
 *
 * Returning null rather than zeros is deliberate: without a torso there is no
 * body frame, and every downstream number would silently be in image
 * coordinates. The segmenter treats null as silence.
 */
export function encodeFrame(raw: RawFrame, dominant: "left" | "right" = "right"): Float32Array | null {
  const bf = bodyFrame(raw.pose);
  if (!bf) return null;

  const out = new Float32Array(STATIC_DIM);
  const mirrored = dominant === "left";
  const dom = mirrored ? raw.leftHand : raw.rightHand;
  const nonDom = mirrored ? raw.rightHand : raw.leftHand;

  encodeHand(dom, bf, out, 0);
  encodeHand(nonDom, bf, out, PER_HAND);

  let o = 2 * PER_HAND;
  for (const idx of POSE_KEEP_ORDER) {
    const b = raw.pose![idx] ? toBody(raw.pose![idx], bf) : [0, 0, 0];
    out[o++] = b[0];
    out[o++] = b[1];
    out[o++] = b[2];
  }
  encodeFace(raw.face, bf, out, o);

  return mirrored ? mirrorStatic(out) : out;
}

/** Append first differences. Movement is a phoneme; handing it to the model
 * directly is free and worth a few points of signer-independent accuracy. */
export function addVelocity(frames: Float32Array[]): Float32Array {
  const T = frames.length;
  const out = new Float32Array(T * FEATURE_DIM);
  for (let t = 0; t < T; t++) {
    const base = t * FEATURE_DIM;
    out.set(frames[t], base);
    if (t > 0) {
      for (let i = 0; i < STATIC_DIM; i++) {
        out[base + STATIC_DIM + i] = frames[t][i] - frames[t - 1][i];
      }
    }
  }
  return out;
}

/** Linear time-warp to a fixed frame count. Discards absolute duration on
 * purpose: signing speed varies hugely between people and is not what
 * distinguishes these signs. */
export function resample(seq: Float32Array, T: number, dim: number, target: number): Float32Array {
  if (T === target) return seq;
  const out = new Float32Array(target * dim);
  if (T === 0) return out;
  for (let i = 0; i < target; i++) {
    const pos = T === 1 ? 0 : (i * (T - 1)) / (target - 1);
    const lo = Math.floor(pos);
    const hi = Math.min(T - 1, lo + 1);
    const w = pos - lo;
    for (let c = 0; c < dim; c++) {
      out[i * dim + c] = seq[lo * dim + c] * (1 - w) + seq[hi * dim + c] * w;
    }
  }
  return out;
}

/** Dominant-wrist motion energy, used by the segmenter. Wrist only: including
 * handshape change makes the trace too noisy for hysteresis to work. */
export function frameEnergy(prev: Float32Array | null, cur: Float32Array): number {
  if (!prev || cur[PER_HAND - 1] < 0.5 || prev[PER_HAND - 1] < 0.5) return 0;
  const o = PER_HAND_SHAPE;
  return Math.hypot(cur[o] - prev[o], cur[o + 1] - prev[o + 1], cur[o + 2] - prev[o + 2]);
}

/** Dominant hand above the hip line. Body-frame y is 0 at the shoulders and
 * ~1 at the hips, so 1.15 is a generous "not resting in your lap". */
export function handsInSigningSpace(f: Float32Array): boolean {
  return f[PER_HAND - 1] >= 0.5 && f[PER_HAND_SHAPE + 1] < 1.15;
}
