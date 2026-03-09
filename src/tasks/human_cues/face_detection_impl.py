"""OpenCV human-cues baseline using a Haar face detector only."""

from src.tasks.interface import TaskResult, make_result


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

    face_detector = getattr(run, "_face_detector", None)

    if face_detector is None:
        face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        run._face_detector = face_detector

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
