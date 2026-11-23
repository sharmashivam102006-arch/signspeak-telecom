"""Dataset collection from a webcam.

Run this on 6-8 different people, in more than one room, at more than one time
of day. The dataset is the bottleneck of this whole project and no architecture
choice will compensate for collecting 300 samples from two flatmates under the
same ceiling light.

    python -m signspeak_ml.capture --subject alice --lighting normal --distance medium
    python -m signspeak_ml.capture --subject bob --dominant left --lighting dim

Controls: SPACE records one sample of the highlighted sign, TAB cycles signs,
N records a NONE (background) sample, R deletes the last take, Q quits.

Collect NONE deliberately, not lazily: scratching, adjusting hair, reaching
off-screen, waving, gesturing while talking, and signs that are NOT in the
vocabulary. The NONE class is what stops the system from confidently
announcing "EMERGENCY" because someone reached for their coffee.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

import numpy as np

from .config import CONFIG, VOCAB
from .features import FEATURE_DIM, add_velocity, encode_frame

try:  # optional at import time so the rest of the package works headless
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:  # pragma: no cover
    cv2 = None
    mp = None


MODEL_URLS = {
    "hand": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "face": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
}


def ensure_models(dest: Path) -> dict[str, Path]:
    """Download the three .task bundles once. MediaPipe Holistic is deprecated;
    the Tasks API landmarkers are the supported path and give better hand
    tracking, at the cost of running three graphs instead of one."""
    import urllib.request

    dest.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, url in MODEL_URLS.items():
        p = dest / f"{name}_landmarker.task"
        if not p.exists():
            print(f"downloading {name} landmarker...")
            urllib.request.urlretrieve(url, p)
        paths[name] = p
    return paths


class LandmarkExtractor:
    def __init__(self, model_dir: str | Path = "models") -> None:
        if cv2 is None:
            raise RuntimeError(
                "capture needs opencv-python and mediapipe:\n"
                "    pip install 'signspeak-ml[capture]'"
            )
        paths = ensure_models(Path(model_dir))
        base = mp_python.BaseOptions

        self.hands = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=base(model_asset_path=str(paths["hand"])),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
        self.pose = mp_vision.PoseLandmarker.create_from_options(
            mp_vision.PoseLandmarkerOptions(
                base_options=base(model_asset_path=str(paths["pose"])),
                running_mode=mp_vision.RunningMode.VIDEO,
            )
        )
        self.face = mp_vision.FaceLandmarker.create_from_options(
            mp_vision.FaceLandmarkerOptions(
                base_options=base(model_asset_path=str(paths["face"])),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_faces=1,
            )
        )

    @staticmethod
    def _to_array(landmarks) -> np.ndarray:
        return np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)

    def process(self, bgr: np.ndarray, ts_ms: int) -> dict:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        hr = self.hands.detect_for_video(image, ts_ms)
        pr = self.pose.detect_for_video(image, ts_ms)
        fr = self.face.detect_for_video(image, ts_ms)

        left = right = None
        for lm, handed in zip(hr.hand_landmarks, hr.handedness):
            arr = self._to_array(lm)
            # MediaPipe labels handedness from the *image's* point of view; the
            # frame is mirrored for the user, so "Left" here is the user's right.
            if handed[0].category_name == "Left":
                right = arr
            else:
                left = arr

        return {
            "pose": self._to_array(pr.pose_landmarks[0]) if pr.pose_landmarks else None,
            "left_hand": left,
            "right_hand": right,
            "face": self._to_array(fr.face_landmarks[0]) if fr.face_landmarks else None,
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True, help="unique id per person - this drives LOSO")
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--lighting", default="normal", choices=["normal", "dim", "backlit", "harsh"])
    ap.add_argument("--distance", default="medium", choices=["near", "medium", "far"])
    ap.add_argument("--background", default="plain", choices=["plain", "cluttered"])
    ap.add_argument("--dominant", default="right", choices=["right", "left"])
    ap.add_argument("--session", default=None)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--max-seconds", type=float, default=3.0)
    args = ap.parse_args()

    session = args.session or time.strftime("%Y%m%d-%H%M")
    out_dir = Path(args.out) / args.subject / session
    out_dir.mkdir(parents=True, exist_ok=True)

    extractor = LandmarkExtractor()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {args.camera}")

    signs = [v for v in VOCAB if v != "NONE"]
    cursor = 0
    recording = False
    buf: list[np.ndarray] = []
    rec_start = 0.0
    last_path: Path | None = None
    counts = {v: 0 for v in VOCAB}
    t0 = time.time()

    print("SPACE record | TAB next sign | N record NONE | R undo | Q quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        ts_ms = int((time.time() - t0) * 1000)
        raw = extractor.process(frame, ts_ms)
        vec = encode_frame(
            raw["pose"], raw["left_hand"], raw["right_hand"], raw["face"], args.dominant
        )

        tracked = vec is not None
        if recording:
            if tracked:
                buf.append(vec)
            if time.time() - rec_start > args.max_seconds:
                recording = False
                last_path = _save(buf, out_dir, pending_label, args, session)
                if last_path:
                    counts[pending_label] += 1
                buf = []

        # --- HUD ---
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 86), (18, 18, 22), -1)
        cv2.putText(frame, f"{signs[cursor]}", (14, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 245), 2)
        cv2.putText(frame, f"subject={args.subject}  light={args.lighting}  "
                           f"dist={args.distance}  dom={args.dominant}",
                    (14, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 160), 1)
        cv2.putText(frame, "  ".join(f"{k}:{v}" for k, v in counts.items() if v),
                    (14, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 190, 140), 1)
        cv2.circle(frame, (w - 28, 28), 10, (0, 200, 0) if tracked else (0, 0, 220), -1)
        if recording:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)
            cv2.putText(frame, f"REC {len(buf)}", (w - 150, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("SignSpeak capture", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == 9:  # TAB
            cursor = (cursor + 1) % len(signs)
        elif key == 32 and not recording:  # SPACE
            pending_label, recording, buf, rec_start = signs[cursor], True, [], time.time()
        elif key == ord("n") and not recording:
            pending_label, recording, buf, rec_start = "NONE", True, [], time.time()
        elif key == ord("r") and last_path and last_path.exists():
            last_path.unlink()
            print(f"deleted {last_path.name}")
            last_path = None

    cap.release()
    cv2.destroyAllWindows()
    print("\nsaved:", json.dumps(counts, indent=2))
    thin = [k for k, v in counts.items() if 0 < v < 15]
    if thin:
        print(f"under 15 samples for: {', '.join(thin)} - aim for 20-30 each per person.")


def _save(buf: list[np.ndarray], out_dir: Path, label: str, args, session: str) -> Path | None:
    if len(buf) < CONFIG.window.min_segment_frames:
        print(f"discarded: only {len(buf)} tracked frames")
        return None
    feats = add_velocity(np.stack(buf, axis=0))
    assert feats.shape[1] == FEATURE_DIM
    path = out_dir / f"{label}_{uuid.uuid4().hex[:8]}.npz"
    np.savez_compressed(
        path,
        features=feats,
        meta=json.dumps(
            {
                "label": label,
                "subject": args.subject,
                "lighting": args.lighting,
                "distance": args.distance,
                "background": args.background,
                "dominant": args.dominant,
                "session": session,
            }
        ),
    )
    print(f"  saved {label}  {feats.shape[0]} frames  -> {path.name}")
    return path


if __name__ == "__main__":
    main()
