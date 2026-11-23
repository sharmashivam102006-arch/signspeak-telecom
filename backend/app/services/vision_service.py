"""MediaPipe Hands landmark extraction — lazy singleton (Tasks API).

MediaPipe >= 1.0 removed the legacy ``mp.solutions.hands`` API, so this
uses the supported Tasks API: a single HandLandmarker created once (on
first use) and reused, guarded by a lock. The model file
(``hand_landmarker.task``) lives in backend/models/; download it with:
    python -m app.services.vision_service --download

If MediaPipe (or OpenCV, or the model file) is not available,
``mediapipe_available()`` is False and extraction raises a clear
RuntimeError — the route turns that into HTTP 503. No fake landmarks
are ever returned.
"""
import os
import threading

from app import config

try:
    import mediapipe as mp  # type: ignore
    from mediapipe.tasks.python.core.base_options import BaseOptions  # type: ignore
    from mediapipe.tasks.python import vision  # type: ignore
    _MP_AVAILABLE = True
except ImportError:
    mp = None
    BaseOptions = None  # type: ignore
    vision = None
    _MP_AVAILABLE = False

HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task")

_landmarker = None
_landmarker_lock = threading.Lock()


def mediapipe_available() -> bool:
    return _MP_AVAILABLE and os.path.isfile(os.path.abspath(config.MEDIAPIPE_MODEL_PATH))


def model_path() -> str:
    return os.path.abspath(config.MEDIAPIPE_MODEL_PATH)


def _get_landmarker():
    """Return the shared HandLandmarker, creating it once."""
    global _landmarker
    if _landmarker is not None:
        return _landmarker
    with _landmarker_lock:
        if _landmarker is None:
            if not _MP_AVAILABLE:
                raise RuntimeError(
                    "MediaPipe is not installed. Install mediapipe and "
                    "opencv-python-headless to enable server-side landmark extraction.")
            path = model_path()
            if not os.path.isfile(path):
                raise RuntimeError(
                    f"MediaPipe hand-landmarker model is not downloaded ({path}). "
                    "Download it with: python -m app.services.vision_service --download")
            options = vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=path),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=1,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            _landmarker = vision.HandLandmarker.create_from_options(options)
        return _landmarker


def extract_hand_landmarks(frame_bgr):
    """Extract 21 hand landmarks from a BGR (OpenCV) frame.

    Returns a list of [x, y, z] (normalised image coords) for the first
    detected hand, or None if no hand is found. Raises RuntimeError when
    MediaPipe/OpenCV/the model file is unavailable.
    """
    if not _MP_AVAILABLE:
        raise RuntimeError(
            "MediaPipe is not installed. Install mediapipe and opencv-python-headless "
            "to enable server-side landmark extraction, or send landmarks from the "
            "browser (MediaPipe JS) to /api/ai/sign/predict.")
    import cv2  # type: ignore

    landmarker = _get_landmarker()
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    with _landmarker_lock:
        result = landmarker.detect(image)
    if not result.hand_landmarks:
        return None
    return [[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[0]]


def release():
    """Close the shared HandLandmarker (tests / shutdown)."""
    global _landmarker
    with _landmarker_lock:
        if _landmarker is not None:
            try:
                _landmarker.close()
            except Exception:
                pass
            _landmarker = None


def download_model() -> str:
    """Download the HandLandmarker .task model. Returns the file path."""
    import urllib.request

    path = model_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    urllib.request.urlretrieve(HAND_LANDMARKER_URL, path)
    return path


if __name__ == "__main__":
    import sys
    if "--download" in sys.argv:
        print("Downloading hand_landmarker.task ->", download_model())
    else:
        print("MediaPipe installed:", _MP_AVAILABLE)
        print("Model present:", mediapipe_available(), model_path())
