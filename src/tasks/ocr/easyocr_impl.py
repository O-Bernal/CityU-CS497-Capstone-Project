"""EasyOCR adapter for saved-image comparison runs."""

from src.tasks.interface import TaskResult, make_result


def _bbox_to_xywh(points: list[list[float]]) -> list[int]:
    """Convert EasyOCR quadrilateral points into x, y, width, height."""
    xs = [int(point[0]) for point in points]
    ys = [int(point[1]) for point in points]
    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def run(frame) -> TaskResult:
    """Process one image and return OCR text plus average confidence."""
    try:
        import easyocr
    except ImportError:
        return make_result(
            task="ocr",
            library="easyocr",
            ok=False,
            error="easyocr is not installed. Install it to run this adapter.",
        )

    reader = getattr(run, "_reader", None)
    if reader is None:
        try:
            reader = easyocr.Reader(["en"], gpu=False)
        except Exception as exc:  # noqa: BLE001
            return make_result(
                task="ocr",
                library="easyocr",
                ok=False,
                error=str(exc),
            )
        run._reader = reader

    try:
        results = reader.readtext(frame, detail=1, paragraph=False)
    except Exception as exc:  # noqa: BLE001
        return make_result(
            task="ocr",
            library="easyocr",
            ok=False,
            error=str(exc),
        )

    texts = []
    confidences = []
    detections = []
    for bbox, text, confidence in results:
        cleaned = str(text).strip()
        conf_value = float(confidence) if confidence is not None else None
        if cleaned:
            texts.append(cleaned)
        if conf_value is not None:
            confidences.append(conf_value)

        detections.append(
            {
                "label": "text",
                "confidence": conf_value,
                "bbox": _bbox_to_xywh(bbox),
            }
        )

    avg_confidence = (sum(confidences) / len(confidences)) if confidences else None

    return make_result(
        task="ocr",
        library="easyocr",
        outputs={
            "text": " ".join(texts).strip(),
            "confidence": avg_confidence,
            "detections": detections,
        },
    )
