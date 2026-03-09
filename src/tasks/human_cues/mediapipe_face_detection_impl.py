"""MediaPipe face-detection adapter for the human-cues comparison task."""

from pathlib import Path

from src.tasks.interface import TaskResult, make_result


def _resolve_tasks_api():
    """Resolve MediaPipe Tasks vision API for face detection."""
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
    except ImportError as exc:
        raise ImportError(
            "mediapipe Tasks vision API is not available. Install a MediaPipe build with mediapipe.tasks.python."
        ) from exc

    face_detector = getattr(vision, "FaceDetector", None)
    face_detector_options = getattr(vision, "FaceDetectorOptions", None)
    running_mode = getattr(vision, "RunningMode", None)
    if any(item is None for item in (face_detector, face_detector_options, running_mode)):
        raise RuntimeError("Installed mediapipe package does not expose the MediaPipe Tasks face detector API.")

    return mp, BaseOptions, face_detector, face_detector_options, running_mode


def _resolve_model_path() -> Path:
    """Resolve the configured MediaPipe face-detector model path."""
    cfg = getattr(run, "_capstone_config", {}) or {}
    mediapipe_cfg = cfg.get("mediapipe", {}) if isinstance(cfg, dict) else {}
    configured = mediapipe_cfg.get("face_detector_model")
    if not configured:
        configured = "models/mediapipe/blaze_face_short_range.tflite"

    model_path = Path(configured)
    if not model_path.is_absolute():
        repo_root = Path(__file__).resolve().parents[3]
        model_path = repo_root / model_path
    return model_path


def run(frame) -> TaskResult:
    """Run MediaPipe face detection and return standardized detections."""
    try:
        mp, BaseOptions, FaceDetector, FaceDetectorOptions, VisionRunningMode = _resolve_tasks_api()
        model_path = _resolve_model_path()
    except (ImportError, RuntimeError) as exc:
        return make_result(
            task="human_cues",
            library="mediapipe",
            ok=False,
            error=str(exc),
        )

    detector = getattr(run, "_detector", None)
    detector_model_path = getattr(run, "_detector_model_path", None)
    if detector is None or detector_model_path != str(model_path):
        if not model_path.exists():
            return make_result(
                task="human_cues",
                library="mediapipe",
                ok=False,
                error=(
                    f"MediaPipe face detector model not found: {model_path}. "
                    "Download a Face Detector .task model and set mediapipe.face_detector_model in config."
                ),
            )

        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionRunningMode.IMAGE,
            min_detection_confidence=0.35,
        )
        detector = FaceDetector.create_from_options(options)
        run._detector = detector
        run._detector_model_path = str(model_path)

    import cv2

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    try:
        results = detector.detect(mp_image)
    except Exception as exc:  # noqa: BLE001
        return make_result(
            task="human_cues",
            library="mediapipe",
            ok=False,
            error=str(exc),
        )

    detections = []
    if getattr(results, "detections", None):
        for det in results.detections:
            bbox = det.bounding_box
            score = det.categories[0].score if getattr(det, "categories", None) else None
            detections.append(
                {
                    "label": "face",
                    "confidence": float(score) if score is not None else None,
                    "bbox": [
                        int(getattr(bbox, "origin_x", 0)),
                        int(getattr(bbox, "origin_y", 0)),
                        int(getattr(bbox, "width", 0)),
                        int(getattr(bbox, "height", 0)),
                    ],
                }
            )

    return make_result(
        task="human_cues",
        library="mediapipe",
        outputs={"detections": detections},
    )
