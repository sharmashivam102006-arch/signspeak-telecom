"""Single source of truth for vocabulary, geometry and windowing.

Anything that has to agree between the Python trainer and the TypeScript
runtime lives here and is emitted to `artifacts/runtime_config.json` at
export time. Drift between the two is the single most common cause of
"works in the notebook, fails in the browser".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------
# NOTE: `NONE` is a real, trained class, not a post-hoc threshold. A closed-set
# classifier over 6 signs will assign one of those 6 labels to a user
# scratching their nose, with high confidence. See docs/evaluation-protocol.md
# section "Open-set behaviour".
VOCAB: list[str] = [
    "NONE",  # background / non-vocabulary movement  (index 0, reserved)
    "HELLO",
    "YES",
    "NO",
    "HELP",
    "THANK_YOU",
    "EMERGENCY",
]

# What the hearing participant actually hears. This is a lookup table, not
# translation, and the write-up must say so.
SPEECH_TEMPLATES: dict[str, str] = {
    "HELLO": "Hello.",
    "YES": "Yes.",
    "NO": "No.",
    "HELP": "I need help.",
    "THANK_YOU": "Thank you.",
    "EMERGENCY": "This is an emergency.",
}

LABEL_TO_INDEX = {label: i for i, label in enumerate(VOCAB)}
NONE_INDEX = LABEL_TO_INDEX["NONE"]
NUM_CLASSES = len(VOCAB)


# --------------------------------------------------------------------------
# Landmark topology (MediaPipe Tasks API)
# --------------------------------------------------------------------------
NUM_HAND_LANDMARKS = 21
NUM_POSE_LANDMARKS = 33

# Pose indices we keep. The legs contribute nothing to a seated/standing
# signer and add variance, so they are dropped.
POSE_KEEP = {
    "nose": 0,
    "left_eye": 2,
    "right_eye": 5,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
}
POSE_KEEP_ORDER = list(POSE_KEEP.values())

# Hand topology shortcuts
WRIST = 0
INDEX_MCP = 5
MIDDLE_MCP = 9
PINKY_MCP = 17


# --------------------------------------------------------------------------
# Windowing / sampling
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class WindowConfig:
    target_fps: int = 25
    # Every segment is resampled to exactly this many frames before it hits
    # the model. Signs vary 0.4s-2.5s; 32 frames @25fps = 1.28s of "sign time"
    # after time-warping, which is enough resolution for the movement phoneme.
    frames: int = 32
    min_segment_frames: int = 8      # < 0.32s: almost certainly not a sign
    max_segment_frames: int = 90     # > 3.6s: almost certainly not one sign


@dataclass(frozen=True)
class SegmenterConfig:
    """Motion-energy segmentation, structurally the same idea as voice
    activity detection. See segment.py for the rationale."""

    enter_threshold: float = 0.035   # normalized units per frame
    exit_threshold: float = 0.018    # hysteresis: exit < enter, avoids chatter
    enter_frames: int = 3            # consecutive frames above enter_threshold
    exit_frames: int = 7             # consecutive frames below exit_threshold
    pad_frames: int = 2              # context kept on each side of a segment
    # Hands must be inside the signing space (above the hip line) or we do not
    # even consider it. Cheap, and removes a lot of gesticulation.
    require_hands_in_signing_space: bool = True


@dataclass(frozen=True)
class DecisionConfig:
    """Two-threshold accept/confirm/reject policy.

    These are placeholders. They MUST be re-fitted on a calibration split with
    `calibrate.py --fit-thresholds`, which picks them from the risk-coverage
    curve for a chosen error budget rather than by eyeballing.
    """

    accept: float = 0.90        # auto-speak
    confirm: float = 0.55       # "Possible: HELLO - repeat to confirm"
    # Below `confirm` -> "Didn't catch that, please repeat."
    consecutive_agreements: int = 1   # raise to 2 for a harsher latency/accuracy trade
    repeat_debounce_ms: int = 1200    # suppress duplicate emission of same label


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 32
    epochs: int = 120
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden: int = 128
    layers: int = 2
    dropout: float = 0.3
    label_smoothing: float = 0.05
    patience: int = 20
    seed: int = 1337
    # Augmentation probabilities - all applied in body-anchored feature space,
    # never on raw pixels (we do not have pixels at training time).
    aug_time_warp: float = 0.5
    aug_jitter: float = 0.5
    aug_rotate: float = 0.3
    aug_scale: float = 0.3
    aug_dropout_hand: float = 0.15


@dataclass(frozen=True)
class Config:
    window: WindowConfig = field(default_factory=WindowConfig)
    segmenter: SegmenterConfig = field(default_factory=SegmenterConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["vocab"] = VOCAB
        d["speech_templates"] = SPEECH_TEMPLATES
        return d


CONFIG = Config()
