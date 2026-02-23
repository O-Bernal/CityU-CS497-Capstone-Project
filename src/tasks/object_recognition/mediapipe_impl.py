"""MediaPipe object-recognition adapter using bundled face detection."""

from src.tasks.interface import TaskResult, make_result


def run(frame) -> TaskResult:
    """Run MediaPipe face detection if available and return standardized detections."""
    try:
        import mediapipe as mp
    except ImportError:
        return make_result(
            task="object_recognition",
            library="mediapipe",
            ok=False,
            error="mediapipe is not installed. Install it to run this adapter.",
        )

    detector = run._detector if hasattr(run, "_detector") else None
    if detector is None:
        detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.35,
        )
        run._detector = detector

    import cv2

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)

    detections = []
    if results.detections:
        h, w = frame.shape[:2]
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            x = max(int(bbox.xmin * w), 0)
            y = max(int(bbox.ymin * h), 0)
            bw = max(int(bbox.width * w), 0)
            bh = max(int(bbox.height * h), 0)
            score = det.score[0] if det.score else None
            detections.append(
                {
                    "label": "face",
                    "confidence": float(score) if score is not None else None,
                    "bbox": [x, y, bw, bh],
                }
            )

    return make_result(
        task="object_recognition",
        library="mediapipe",
        outputs={"detections": detections},
    )
