"""OpenCV object-recognition adapter using classical Haar cascades."""

from src.tasks.interface import TaskResult, make_result


def _as_detection(label: str, x: int, y: int, w: int, h: int) -> dict:
    """Convert OpenCV box coordinates to shared detection schema."""
    return {
        "label": label,
        "confidence": None,
        "bbox": [int(x), int(y), int(w), int(h)],
    }


def run(frame) -> TaskResult:
    """Detect face/body regions and return standardized detections."""
    import cv2

    face_detector = run._face_detector if hasattr(run, "_face_detector") else None
    body_detector = run._body_detector if hasattr(run, "_body_detector") else None

    if face_detector is None or body_detector is None:
        face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        body_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")
        run._face_detector = face_detector
        run._body_detector = body_detector

    if face_detector.empty() or body_detector.empty():
        return make_result(
            task="object_recognition",
            library="opencv",
            ok=False,
            error="OpenCV Haar cascades failed to load.",
        )

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    bodies = body_detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3)

    detections = []
    for (x, y, w, h) in faces:
        detections.append(_as_detection("face", x, y, w, h))
    for (x, y, w, h) in bodies:
        detections.append(_as_detection("person", x, y, w, h))

    return make_result(
        task="object_recognition",
        library="opencv",
        outputs={"detections": detections},
    )
