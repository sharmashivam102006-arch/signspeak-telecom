"""Landmark -> feature vector.

This file matters more than the choice of model. MediaPipe emits landmarks in
*image-normalized* coordinates, so the same sign performed by the same person
one step closer to the camera, or framed slightly left of centre, produces a
numerically different sequence. A model trained on raw landmarks learns the
recording setup, not the sign, and then collapses on anyone new.

Everything here exists to strip out the nuisance variation and keep the four
things sign linguistics says actually carry meaning:

    handshape   - finger configuration, independent of where the hand is
    location    - where the hand sits relative to the body
    movement    - how location and shape change over time
    orientation - which way the palm faces

plus a small set of non-manual markers (brow, mouth, head) because some signs
are distinguished only by them.

The reference implementation is mirrored, function for function, in
`apps/web/lib/features.ts`. If you change one, change both, and run
`ml/scripts/check_parity.py`, which asserts the two produce identical vectors
for the same input.
"""

from __future__ import annotations

import numpy as np

from .config import (
    INDEX_MCP,
    MIDDLE_MCP,
    NUM_HAND_LANDMARKS,
    PINKY_MCP,
    POSE_KEEP,
    POSE_KEEP_ORDER,
    WRIST,
)

EPS = 1e-6

# Non-manual signals we extract from the face mesh, by MediaPipe FaceLandmarker
# index. We deliberately keep this tiny: full face mesh is 478 points of mostly
# identity information, which is both privacy-hostile and a gift to a model
# looking for a shortcut to overfit on.
FACE_POINTS = {
    "brow_left": 105,
    "brow_right": 334,
    "eye_left_top": 159,
    "eye_left_bottom": 145,
    "mouth_top": 13,
    "mouth_bottom": 14,
    "mouth_left": 61,
    "mouth_right": 291,
    "nose_tip": 1,
    "chin": 152,
}

# Feature block sizes, used for assertions and for the TS parity check.
PER_HAND_SHAPE = NUM_HAND_LANDMARKS * 3        # 63 - wrist-relative, size-normalized
PER_HAND_POSITION = 3                          # wrist in body frame
PER_HAND_ORIENTATION = 6                       # palm normal + finger axis
PER_HAND_PRESENCE = 1
PER_HAND = PER_HAND_SHAPE + PER_HAND_POSITION + PER_HAND_ORIENTATION + PER_HAND_PRESENCE  # 73
POSE_BLOCK = len(POSE_KEEP_ORDER) * 3          # 33
FACE_BLOCK = 5                                 # brow raise, eye open, mouth h/w, head yaw
STATIC_DIM = 2 * PER_HAND + POSE_BLOCK + FACE_BLOCK   # 184
FEATURE_DIM = STATIC_DIM * 2                   # + first differences (velocity) = 368


# ---------------------------------------------------------------------------
# Body-anchored reference frame
# ---------------------------------------------------------------------------
def body_frame(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Build an orthonormal frame anchored on the signer's torso.

    Returns (origin, R, scale) where R has the body axes as *rows*, so
    ``R @ (p - origin) / scale`` maps a world point into body coordinates.

    origin  mid-shoulder
    x       right shoulder -> left shoulder (signer's own left)
    y       shoulders -> hips (down the torso)
    z       x cross y (out of the chest, toward the camera)
    scale   shoulder width

    Anchoring on the torso rather than the image removes camera distance,
    framing, and roll in one step. It does not remove yaw (signer turned away
    from the camera) - that is handled by augmentation, and by the robustness
    slice in the evaluation protocol.
    """
    if pose is None or pose.shape[0] < 25:
        return None

    ls = pose[POSE_KEEP["left_shoulder"]][:3]
    rs = pose[POSE_KEEP["right_shoulder"]][:3]
    lh = pose[POSE_KEEP["left_hip"]][:3]
    rh = pose[POSE_KEEP["right_hip"]][:3]

    if not np.all(np.isfinite([ls, rs, lh, rh])):
        return None

    origin = (ls + rs) / 2.0
    hip_mid = (lh + rh) / 2.0

    x_axis = ls - rs
    scale = float(np.linalg.norm(x_axis))
    if scale < EPS:
        return None
    x_axis = x_axis / scale

    y_raw = hip_mid - origin
    # Gram-Schmidt: remove the component of y along x so the frame is orthonormal.
    y_axis = y_raw - np.dot(y_raw, x_axis) * x_axis
    ny = np.linalg.norm(y_axis)
    if ny < EPS:
        # Degenerate (hips not detected sensibly): fall back to image-down.
        y_axis = np.array([0.0, 1.0, 0.0])
        y_axis = y_axis - np.dot(y_axis, x_axis) * x_axis
        ny = np.linalg.norm(y_axis)
        if ny < EPS:
            return None
    y_axis = y_axis / ny

    z_axis = np.cross(x_axis, y_axis)
    R = np.stack([x_axis, y_axis, z_axis], axis=0)
    return origin, R, scale


def to_body(points: np.ndarray, origin: np.ndarray, R: np.ndarray, scale: float) -> np.ndarray:
    """Map (N,3) world points into the body frame."""
    return (points[:, :3] - origin) @ R.T / scale


# ---------------------------------------------------------------------------
# Per-hand features
# ---------------------------------------------------------------------------
def hand_shape(hand_body: np.ndarray) -> np.ndarray:
    """Handshape, stripped of where the hand is and how big it is.

    Translate to the wrist, then divide by the wrist->middle-MCP distance. That
    distance is the most stable proxy for hand size available from landmarks
    alone, so an adult and a child making the same handshape land in the same
    place in feature space.
    """
    rel = hand_body - hand_body[WRIST]
    size = float(np.linalg.norm(rel[MIDDLE_MCP]))
    if size < EPS:
        size = 1.0
    return (rel / size).reshape(-1)


def hand_orientation(hand_body: np.ndarray) -> np.ndarray:
    """Palm normal + finger axis, as unit vectors in the body frame.

    Orientation is a genuine phoneme: in many sign languages the same handshape
    and location with the palm flipped is a different sign entirely.
    """
    wrist = hand_body[WRIST]
    v1 = hand_body[INDEX_MCP] - wrist
    v2 = hand_body[PINKY_MCP] - wrist

    normal = np.cross(v1, v2)
    n = np.linalg.norm(normal)
    normal = normal / n if n > EPS else np.zeros(3)

    finger = hand_body[MIDDLE_MCP] - wrist
    f = np.linalg.norm(finger)
    finger = finger / f if f > EPS else np.zeros(3)

    return np.concatenate([normal, finger])


def encode_hand(hand: np.ndarray | None, origin, R, scale) -> np.ndarray:
    """One hand -> 73 floats. Absent hand -> zeros with the presence flag at 0.

    The presence flag is not optional. Without it, "no hand detected" and "hand
    exactly at the body origin" are the same vector, and the model will happily
    learn that dropped tracking means HELP because that is the class where
    tracking dropped most often in your dataset.
    """
    if hand is None or hand.shape[0] != NUM_HAND_LANDMARKS:
        return np.zeros(PER_HAND, dtype=np.float32)

    hb = to_body(hand, origin, R, scale)
    return np.concatenate(
        [
            hand_shape(hb),           # 63
            hb[WRIST],                # 3  - location in body frame
            hand_orientation(hb),     # 6
            np.array([1.0]),          # 1  - present
        ]
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Non-manual markers
# ---------------------------------------------------------------------------
def encode_face(face: np.ndarray | None, scale: float) -> np.ndarray:
    """Five scalars: brow raise, eye aperture, mouth height, mouth width, head yaw.

    Five, not 478. This is enough to separate e.g. a plain statement from a
    question marker, and it is the difference between storing a facial
    expression and storing a face.
    """
    if face is None or face.shape[0] <= max(FACE_POINTS.values()):
        return np.zeros(FACE_BLOCK, dtype=np.float32)

    p = {k: face[i][:3] for k, i in FACE_POINTS.items()}
    inv = 1.0 / max(scale, EPS)

    eye_mid_y = (p["eye_left_top"][1] + p["eye_left_bottom"][1]) / 2.0
    brow_raise = (eye_mid_y - (p["brow_left"][1] + p["brow_right"][1]) / 2.0) * inv
    eye_open = abs(p["eye_left_top"][1] - p["eye_left_bottom"][1]) * inv
    mouth_h = abs(p["mouth_top"][1] - p["mouth_bottom"][1]) * inv
    mouth_w = abs(p["mouth_left"][0] - p["mouth_right"][0]) * inv

    # Head yaw proxy: nose tip offset from the mouth-corner midpoint, along x.
    mouth_mid_x = (p["mouth_left"][0] + p["mouth_right"][0]) / 2.0
    head_yaw = (p["nose_tip"][0] - mouth_mid_x) * inv

    return np.array([brow_raise, eye_open, mouth_h, mouth_w, head_yaw], dtype=np.float32)


# ---------------------------------------------------------------------------
# Handedness canonicalization
# ---------------------------------------------------------------------------
def canonicalize_handedness(
    left: np.ndarray | None,
    right: np.ndarray | None,
    dominant: str = "right",
) -> tuple[np.ndarray | None, np.ndarray | None, bool]:
    """Return (dominant_hand, non_dominant_hand, mirrored).

    Roughly one signer in ten is left-dominant. To a model fed raw left/right
    channels, a left-dominant HELLO is a different sign from a right-dominant
    HELLO, so it needs separate training data for both - which you will not
    have. Instead we mirror left-dominant signers into the right-dominant
    convention and treat handedness as a nuisance variable.

    Mirroring is done in the body frame downstream by negating the x axis; here
    we only decide which channel is which.
    """
    if dominant == "left":
        return left, right, True
    return right, left, False


def mirror_body_features(vec: np.ndarray) -> np.ndarray:
    """Negate the body-frame x component of every 3-vector in a static block.

    Applied to a left-dominant signer's features this yields the right-dominant
    equivalent. Also used as a training augmentation.
    """
    v = vec.copy()
    o = 0
    for _ in range(2):  # two hands
        shape = v[o : o + PER_HAND_SHAPE].reshape(-1, 3)
        shape[:, 0] *= -1
        v[o : o + PER_HAND_SHAPE] = shape.reshape(-1)
        o += PER_HAND_SHAPE
        v[o] *= -1                       # wrist x
        o += PER_HAND_POSITION
        v[o] *= -1                       # palm normal x
        v[o + 3] *= -1                   # finger axis x
        o += PER_HAND_ORIENTATION
        o += PER_HAND_PRESENCE
    pose = v[o : o + POSE_BLOCK].reshape(-1, 3)
    pose[:, 0] *= -1
    v[o : o + POSE_BLOCK] = pose.reshape(-1)
    o += POSE_BLOCK
    v[o + 4] *= -1                       # head yaw
    return v


# ---------------------------------------------------------------------------
# Frame and sequence encoding
# ---------------------------------------------------------------------------
def encode_frame(
    pose: np.ndarray | None,
    left_hand: np.ndarray | None,
    right_hand: np.ndarray | None,
    face: np.ndarray | None = None,
    dominant: str = "right",
) -> np.ndarray | None:
    """One frame of raw MediaPipe output -> STATIC_DIM floats, or None.

    Returns None when the torso is not visible, because without a torso there
    is no body frame and every downstream number would be in image coordinates
    - silently, which is worse than failing.
    """
    bf = body_frame(pose) if pose is not None else None
    if bf is None:
        return None
    origin, R, scale = bf

    dom, non_dom, mirrored = canonicalize_handedness(left_hand, right_hand, dominant)

    vec = np.concatenate(
        [
            encode_hand(dom, origin, R, scale),
            encode_hand(non_dom, origin, R, scale),
            to_body(pose[POSE_KEEP_ORDER], origin, R, scale).reshape(-1),
            encode_face(face, scale),
        ]
    ).astype(np.float32)

    if mirrored:
        vec = mirror_body_features(vec)

    assert vec.shape[0] == STATIC_DIM, (vec.shape, STATIC_DIM)
    return vec


def add_velocity(seq: np.ndarray) -> np.ndarray:
    """(T, STATIC_DIM) -> (T, FEATURE_DIM) by appending first differences.

    Movement is a phoneme, not a nuisance. A GRU can in principle infer
    velocity itself, but handing it to the model directly is free and measurably
    worth ~3-5 points of signer-independent accuracy in this feature space.
    """
    if seq.shape[0] == 0:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32)
    vel = np.zeros_like(seq)
    vel[1:] = seq[1:] - seq[:-1]
    return np.concatenate([seq, vel], axis=1).astype(np.float32)


def resample(seq: np.ndarray, frames: int) -> np.ndarray:
    """Linear time-warp to a fixed number of frames.

    Uniform resampling deliberately discards absolute duration. Signing speed
    varies enormously between people and is not what distinguishes HELLO from
    THANK_YOU, so normalizing it away is the right call here. If you later add
    signs that differ only in duration, revisit this - and say so in the report.
    """
    T = seq.shape[0]
    if T == 0:
        return np.zeros((frames, seq.shape[1] if seq.ndim > 1 else FEATURE_DIM), dtype=np.float32)
    if T == frames:
        return seq.astype(np.float32)
    src = np.linspace(0.0, T - 1.0, num=T)
    dst = np.linspace(0.0, T - 1.0, num=frames)
    out = np.empty((frames, seq.shape[1]), dtype=np.float32)
    for c in range(seq.shape[1]):
        out[:, c] = np.interp(dst, src, seq[:, c])
    return out


def encode_sequence(
    frames_raw: list[dict],
    target_frames: int,
    dominant: str = "right",
) -> np.ndarray | None:
    """List of per-frame landmark dicts -> (target_frames, FEATURE_DIM).

    Each dict: {"pose": (33,3|4), "left_hand": (21,3)|None,
                "right_hand": (21,3)|None, "face": (478,3)|None}

    Frames with no usable torso are dropped rather than zero-filled; if more
    than half of them are unusable the whole sample is rejected, because a
    half-tracked sample is training-set poison.
    """
    encoded = []
    for f in frames_raw:
        v = encode_frame(
            f.get("pose"), f.get("left_hand"), f.get("right_hand"), f.get("face"), dominant
        )
        if v is not None:
            encoded.append(v)

    if not encoded or len(encoded) < 0.5 * max(len(frames_raw), 1):
        return None

    static = np.stack(encoded, axis=0)
    return resample(add_velocity(static), target_frames)


def motion_energy(seq_static: np.ndarray) -> np.ndarray:
    """Per-frame dominant-hand motion energy, used by the segmenter.

    Computed only from the dominant wrist position in the body frame, because
    that is the signal that actually starts and stops with a sign. Including
    handshape change here makes the energy trace noisy in a way that ruins
    hysteresis.
    """
    if seq_static.shape[0] < 2:
        return np.zeros(seq_static.shape[0], dtype=np.float32)
    wrist_off = PER_HAND_SHAPE
    wrist = seq_static[:, wrist_off : wrist_off + 3]
    d = np.zeros(seq_static.shape[0], dtype=np.float32)
    d[1:] = np.linalg.norm(wrist[1:] - wrist[:-1], axis=1)
    return d
