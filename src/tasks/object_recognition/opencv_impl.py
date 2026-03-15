"""OpenCV object-recognition adapter using the DNN detection API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.tasks.interface import TaskResult, make_result


DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "opencv" / "frozen_inference_graph.pb"
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "models" / "opencv" / "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
)
DEFAULT_LABELS = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

_CONFIG: dict[str, Any] = {}
_DETECTOR: Any | None = None
_DETECTOR_KEY: tuple[str, str] | None = None


def configure(cfg: dict[str, Any]) -> None:
    """Store run-scoped config for later detector resolution."""
    global _CONFIG
    _CONFIG = cfg


def _resolve_dnn_config() -> tuple[Path, Path, set[str] | None, float, float, int, float]:
    """Resolve model, config, and optional label filter paths from config."""
    cfg = _CONFIG or {}
    dnn_cfg = cfg.get("opencv_dnn", {}) if isinstance(cfg, dict) else {}

    model_path = Path(dnn_cfg.get("model", DEFAULT_MODEL_PATH))
    config_path = Path(dnn_cfg.get("config", DEFAULT_CONFIG_PATH))
    if not model_path.is_absolute():
        model_path = Path(__file__).resolve().parents[3] / model_path
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parents[3] / config_path

    target_labels = dnn_cfg.get("target_labels")
    label_filter = {str(label).strip().lower() for label in target_labels} if target_labels else None
    filter_enabled = bool(dnn_cfg.get("enforce_target_labels", False))
    confidence_threshold = float(dnn_cfg.get("confidence_threshold", 0.15))
    nms_threshold = float(dnn_cfg.get("nms_threshold", 0.4))
    input_size = int(dnn_cfg.get("input_size", 320))
    center_crop_fraction = float(dnn_cfg.get("center_crop_fraction", 0.7))
    return (
        model_path,
        config_path,
        (label_filter if filter_enabled else None),
        confidence_threshold,
        nms_threshold,
        input_size,
        center_crop_fraction,
    )


def _class_name_for_id(class_id: int) -> str:
    """Map a 1-based COCO class id to a display label."""
    index = int(class_id) - 1
    if 0 <= index < len(DEFAULT_LABELS):
        return DEFAULT_LABELS[index]
    return f"class_{class_id}"


def _normalize_label(label: str) -> str:
    """Normalize detector labels to the shared comparison vocabulary."""
    lowered = str(label).strip().lower()
    if lowered == "tv":
        return "tvmonitor"
    return lowered


def _create_detection_model(cv2_module: Any, model_path: Path, config_path: Path) -> Any:
    """Create an OpenCV DNN detection model across stub/runtime variants."""
    model_factory = getattr(cv2_module, "dnn_DetectionModel", None)
    if model_factory is None:
        dnn_module = getattr(cv2_module, "dnn", None)
        model_factory = getattr(dnn_module, "DetectionModel", None) if dnn_module is not None else None
    if model_factory is None:
        raise RuntimeError("OpenCV build does not expose a DNN DetectionModel constructor.")
    return model_factory(str(model_path), str(config_path))


def _build_candidate_views(frame, *, center_crop_fraction: float) -> list[dict]:
    """Build full-frame and center-crop views for detector retries."""
    height, width = frame.shape[:2]
    views = [{"image": frame, "offset": (0, 0)}]

    crop_width = max(1, int(width * center_crop_fraction))
    crop_height = max(1, int(height * center_crop_fraction))
    x = max(0, (width - crop_width) // 2)
    y = max(0, (height - crop_height) // 2)
    cropped = frame[y : y + crop_height, x : x + crop_width]
    views.insert(0, {"image": cropped, "offset": (x, y)})
    return views


def run(frame) -> TaskResult:
    """Detect objects in the webcam frame using OpenCV DNN."""
    import cv2

    global _DETECTOR, _DETECTOR_KEY
    (
        model_path,
        config_path,
        label_filter,
        confidence_threshold,
        nms_threshold,
        input_size,
        center_crop_fraction,
    ) = _resolve_dnn_config()
    detector = _DETECTOR
    detector_key = _DETECTOR_KEY
    expected_key = (str(model_path), str(config_path))

    if detector is None or detector_key != expected_key:
        if not model_path.exists():
            return make_result(
                task="object_recognition",
                library="opencv",
                ok=False,
                error=f"OpenCV DNN model not found: {model_path}",
            )
        if not config_path.exists():
            return make_result(
                task="object_recognition",
                library="opencv",
                ok=False,
                error=f"OpenCV DNN config not found: {config_path}",
            )

        try:
            net = _create_detection_model(cv2, model_path, config_path)
            net.setInputSize(input_size, input_size)
            net.setInputScale(1.0 / 127.5)
            net.setInputMean((127.5, 127.5, 127.5))
            net.setInputSwapRB(True)
        except Exception as exc:  # noqa: BLE001
            return make_result(
                task="object_recognition",
                library="opencv",
                ok=False,
                error=str(exc),
            )

        _DETECTOR = net
        _DETECTOR_KEY = expected_key
        detector = net

    detections = []
    scores: dict[str, float] = {}
    best_label = None
    best_score = 0.0

    try:
        for view in _build_candidate_views(frame, center_crop_fraction=center_crop_fraction):
            class_ids, confidences, boxes = detector.detect(
                view["image"],
                confThreshold=confidence_threshold,
                nmsThreshold=nms_threshold,
            )

            if class_ids is None or len(class_ids) == 0:
                continue

            x_offset, y_offset = view["offset"]
            for class_id, confidence, box in zip(class_ids.flatten(), confidences.flatten(), boxes):
                label = _normalize_label(_class_name_for_id(int(class_id)))
                score = float(confidence)

                prior = scores.get(label, 0.0)
                if score > prior:
                    scores[label] = round(score, 4)

                if label_filter and label.lower() not in label_filter:
                    continue

                adjusted_box = [
                    int(box[0] + x_offset),
                    int(box[1] + y_offset),
                    int(box[2]),
                    int(box[3]),
                ]
                detections.append(
                    {
                        "label": label,
                        "confidence": score,
                        "bbox": adjusted_box,
                    }
                )
                if score > best_score:
                    best_score = score
                    best_label = label
    except Exception as exc:  # noqa: BLE001
        return make_result(
            task="object_recognition",
            library="opencv",
            ok=False,
            error=str(exc),
        )

    return make_result(
        task="object_recognition",
        library="opencv",
        outputs={
            "detections": detections,
            "scores": scores,
            "matched_label": best_label,
            "candidate_labels": list(scores.keys()),
        },
    )
