"""Placeholder Tesseract OCR task interface."""

from src.tasks.interface import TaskResult, make_result


def run(frame) -> TaskResult:
    """Process one frame and return OCR text output."""
    # Placeholder implementation for OCR pipeline.
    return make_result(
        task="ocr",
        library="tesseract",
        outputs={"text": ""},
    )
