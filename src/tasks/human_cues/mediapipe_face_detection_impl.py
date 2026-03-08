"""MediaPipe face-detection adapter for the human-cues comparison task."""

from src.tasks.interface import TaskResult, make_result


def run(frame) -> TaskResult:
    """Run MediaPipe face detection and return standardized detections."""
    try:
        import mediapipe as mp
    except ImportError:
        return make_result(
            task="human_cues",
            library="mediapipe",
            ok=False,
            error="mediapipe is not installed. Install it to run this adapter.",
        )

    detector = getattr(run, "_detector", None)
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
        frame_height, frame_width = frame.shape[:2]
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            x = max(int(bbox.xmin * frame_width), 0)
            y = max(int(bbox.ymin * frame_height), 0)
            width = max(int(bbox.width * frame_width), 0)
            height = max(int(bbox.height * frame_height), 0)
            score = det.score[0] if det.score else None
            detections.append(
                {
                    "label": "face",
                    "confidence": float(score) if score is not None else None,
                    "bbox": [x, y, width, height],
                }
            )

    return make_result(
        task="human_cues",
        library="mediapipe",
        outputs={"detections": detections},
    )
