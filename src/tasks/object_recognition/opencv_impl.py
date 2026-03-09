"""OpenCV object-recognition adapter using the DNN detection API."""

from __future__ import annotations

from pathlib import Path

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


def _resolve_dnn_config() -> tuple[Path, Path, set[str] | None]:
    """Resolve model, config, and optional label filter paths from config."""
    cfg = getattr(run, "_capstone_config", {}) or {}
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
    return model_path, config_path, (label_filter if filter_enabled else None), confidence_threshold, nms_threshold


def _class_name_for_id(class_id: int) -> str:
    """Map a 1-based COCO class id to a display label."""
    index = int(class_id) - 1
    if 0 <= index < len(DEFAULT_LABELS):
        return DEFAULT_LABELS[index]
    return f"class_{class_id}"


def run(frame) -> TaskResult:
    """Detect objects in the webcam frame using OpenCV DNN."""
    import cv2

    model_path, config_path, label_filter, confidence_threshold, nms_threshold = _resolve_dnn_config()
    detector = getattr(run, "_detector", None)
    detector_key = getattr(run, "_detector_key", None)
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
            net = cv2.dnn_DetectionModel(str(model_path), str(config_path))
            net.setInputSize(320, 320)
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

        run._detector = net
        run._detector_key = expected_key
        detector = net

    try:
        class_ids, confidences, boxes = detector.detect(
            frame,
            confThreshold=confidence_threshold,
            nmsThreshold=nms_threshold,
        )
    except Exception as exc:  # noqa: BLE001
        return make_result(
            task="object_recognition",
            library="opencv",
            ok=False,
            error=str(exc),
        )

    detections = []
    scores: dict[str, float] = {}
    best_label = None
    best_score = 0.0

    if class_ids is not None and len(class_ids) > 0:
        for class_id, confidence, box in zip(class_ids.flatten(), confidences.flatten(), boxes):
            label = _class_name_for_id(int(class_id))
            if label_filter and label.lower() not in label_filter:
                continue

            score = float(confidence)
            detections.append(
                {
                    "label": label,
                    "confidence": score,
                    "bbox": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                }
            )
            prior = scores.get(label, 0.0)
            if score > prior:
                scores[label] = round(score, 4)
            if score > best_score:
                best_score = score
                best_label = label

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
