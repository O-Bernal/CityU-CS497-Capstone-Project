"""OpenCV human-cues baseline using a Haar face detector only."""

from pathlib import Path
from typing import Any

from src.tasks.interface import TaskResult, make_result

_FACE_DETECTOR: Any | None = None
_CASCADE_FILENAME = "haarcascade_frontalface_default.xml"


def _resolve_cascade_path(cv2_module: Any) -> str:
    """Resolve the Haar cascade path across OpenCV stub variants."""
    data_dir = getattr(cv2_module, "data", None)
    haarcascades = getattr(data_dir, "haarcascades", None) if data_dir is not None else None
    if haarcascades:
        return str(Path(haarcascades) / _CASCADE_FILENAME)

    package_dir = Path(cv2_module.__file__).resolve().parent
    fallback = package_dir / "data" / _CASCADE_FILENAME
    return str(fallback)


def _as_detection(label: str, x: int, y: int, w: int, h: int) -> dict:
    """Convert OpenCV box coordinates to the shared detection schema."""
    return {
        "label": label,
        "confidence": None,
        "bbox": [int(x), int(y), int(w), int(h)],
    }


def run(frame) -> TaskResult:
    """Detect face regions and return standardized detections."""
    import cv2

    global _FACE_DETECTOR
    face_detector = _FACE_DETECTOR

    if face_detector is None:
        face_detector = cv2.CascadeClassifier(_resolve_cascade_path(cv2))
        _FACE_DETECTOR = face_detector

    if face_detector.empty():
        return make_result(
            task="human_cues",
            library="opencv",
            ok=False,
            error="OpenCV Haar face cascade failed to load.",
        )

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    detections = []
    for (x, y, w, h) in faces:
        detections.append(_as_detection("face", x, y, w, h))

    return make_result(
        task="human_cues",
        library="opencv",
        outputs={"detections": detections},
    )
