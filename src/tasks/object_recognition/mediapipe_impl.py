"""MediaPipe object-recognition adapter using the Tasks object detector API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.tasks.interface import TaskResult, make_result

DEFAULT_OBJECT_MODEL = Path(__file__).resolve().parents[3] / "models" / "mediapipe" / "efficientdet_lite0.tflite"

_CONFIG: dict[str, Any] = {}
_DETECTOR: Any | None = None
_DETECTOR_MODEL_PATH: str | None = None


def configure(cfg: dict[str, Any]) -> None:
    """Store run-scoped config for later model resolution."""
    global _CONFIG
    _CONFIG = cfg


def _resolve_model_path() -> Path:
    """Resolve the configured MediaPipe object-detector model path."""
    cfg = _CONFIG or {}
    mediapipe_cfg = cfg.get("mediapipe", {}) if isinstance(cfg, dict) else {}
    configured = (
        mediapipe_cfg.get("object_detector_model")
        or mediapipe_cfg.get("image_embedder_model")
    )
    model_path = Path(configured) if configured else DEFAULT_OBJECT_MODEL
    if not model_path.is_absolute():
        model_path = Path(__file__).resolve().parents[3] / model_path
    return model_path


def _resolve_label_filter() -> set[str] | None:
    """Resolve optional target-label filtering from config."""
    cfg = _CONFIG or {}
    mediapipe_cfg = cfg.get("mediapipe", {}) if isinstance(cfg, dict) else {}
    labels = mediapipe_cfg.get("target_labels")
    if not labels:
        return None
    if not mediapipe_cfg.get("enforce_target_labels", True):
        return None
    return {str(label).strip().lower() for label in labels}


def _normalize_label(label: str) -> str:
    """Normalize library-specific labels to the shared comparison vocabulary."""
    lowered = str(label).strip().lower()
    if lowered == "tv":
        return "tvmonitor"
    return lowered


def run(frame) -> TaskResult:
    """Detect objects in the webcam frame using MediaPipe Tasks."""
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
    except ImportError:
        return make_result(
            task="object_recognition",
            library="mediapipe",
            ok=False,
            error="mediapipe Tasks vision API is not available for object recognition.",
        )

    model_path = _resolve_model_path()
    label_filter = _resolve_label_filter()
    global _DETECTOR, _DETECTOR_MODEL_PATH
    detector = _DETECTOR
    detector_model_path = _DETECTOR_MODEL_PATH
    if detector is None or detector_model_path != str(model_path):
        if not model_path.exists():
            return make_result(
                task="object_recognition",
                library="mediapipe",
                ok=False,
                error=(
                    f"MediaPipe object detector model not found: {model_path}. "
                    "Set mediapipe.object_detector_model to a compatible .tflite file."
                ),
            )

        options = vision.ObjectDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            score_threshold=0.25,
            max_results=5,
        )
        detector = vision.ObjectDetector.create_from_options(options)
        _DETECTOR = detector
        _DETECTOR_MODEL_PATH = str(model_path)

    import cv2

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    try:
        results = detector.detect(mp_image)
    except Exception as exc:  # noqa: BLE001
        return make_result(
            task="object_recognition",
            library="mediapipe",
            ok=False,
            error=str(exc),
        )

    detections = []
    best_label = None
    best_score = -1.0
    scores: dict[str, float] = {}

    for det in getattr(results, "detections", []) or []:
        bbox = det.bounding_box
        category = det.categories[0] if getattr(det, "categories", None) else None
        raw_label = getattr(category, "category_name", None) or getattr(category, "display_name", None) or "object"
        label = _normalize_label(raw_label)
        if label_filter and label not in label_filter:
            continue
        score = float(category.score) if category is not None and getattr(category, "score", None) is not None else None
        if score is not None:
            prior = scores.get(label, 0.0)
            if score > prior:
                scores[label] = round(score, 4)
            if score > best_score:
                best_score = score
                best_label = label

        detections.append(
            {
                "label": label,
                "confidence": score,
                "bbox": [
                    int(getattr(bbox, "origin_x", 0)),
                    int(getattr(bbox, "origin_y", 0)),
                    int(getattr(bbox, "width", 0)),
                    int(getattr(bbox, "height", 0)),
                ],
            }
        )

    return make_result(
        task="object_recognition",
        library="mediapipe",
        outputs={
            "detections": detections,
            "scores": scores,
            "matched_label": best_label,
        },
    )
