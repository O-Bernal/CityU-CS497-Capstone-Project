"""Tesseract OCR adapter for saved-image comparison runs."""

import os
from pathlib import Path
import shutil

from src.tasks.interface import TaskResult, make_result


WINDOWS_TESSERACT_CANDIDATES = [
    Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
    Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
]


def _resolve_tesseract_cmd() -> str | None:
    """Resolve the Tesseract executable from env, PATH, or common Windows paths."""
    configured = os.getenv("TESSERACT_CMD")
    if configured:
        return configured

    discovered = shutil.which("tesseract")
    if discovered:
        return discovered

    for candidate in WINDOWS_TESSERACT_CANDIDATES:
        if candidate.exists():
            return str(candidate)

    return None


def _configure_tesseract(pytesseract_module) -> str | None:
    """Point pytesseract at a resolved Tesseract executable, if one can be found."""
    tesseract_cmd = _resolve_tesseract_cmd()
    if tesseract_cmd:
        pytesseract_module.pytesseract.tesseract_cmd = tesseract_cmd
    return tesseract_cmd


def run(frame) -> TaskResult:
    """Process one image and return OCR text plus average confidence."""
    try:
        import pytesseract
    except ImportError:
        return make_result(
            task="ocr",
            library="tesseract",
            ok=False,
            error="pytesseract is not installed. Install it to run this adapter.",
        )

    import cv2

    configured_cmd = _configure_tesseract(pytesseract)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

    try:
        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    except pytesseract.pytesseract.TesseractNotFoundError:
        detail = (
            f"Configured path: {configured_cmd}" if configured_cmd else "Checked PATH and common Windows install locations."
        )
        return make_result(
            task="ocr",
            library="tesseract",
            ok=False,
            error=(
                "Tesseract executable not found. Install Tesseract OCR and either add it to PATH, "
                "set TESSERACT_CMD, or set ocr.tesseract_cmd in configs/task_ocr.yaml. "
                f"{detail}"
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return make_result(
            task="ocr",
            library="tesseract",
            ok=False,
            error=str(exc),
        )

    texts = []
    confidences = []
    detections = []
    for idx, raw_text in enumerate(data.get("text", [])):
        text = str(raw_text).strip()
        conf_raw = data.get("conf", ["-1"])[idx]
        try:
            conf_value = float(conf_raw)
        except (TypeError, ValueError):
            conf_value = -1.0

        confidence = None if conf_value < 0 else round(conf_value / 100.0, 4)
        if not text:
            continue

        texts.append(text)
        if confidence is not None:
            confidences.append(confidence)

        left = int(data.get("left", [0])[idx])
        top = int(data.get("top", [0])[idx])
        width = int(data.get("width", [0])[idx])
        height = int(data.get("height", [0])[idx])
        detections.append(
            {
                "label": "text",
                "confidence": confidence,
                "bbox": [left, top, width, height],
            }
        )

    avg_confidence = (sum(confidences) / len(confidences)) if confidences else None

    return make_result(
        task="ocr",
        library="tesseract",
        outputs={
            "text": " ".join(texts).strip(),
            "confidence": avg_confidence,
            "detections": detections,
        },
    )
