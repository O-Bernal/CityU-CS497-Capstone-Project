"""Tests for OCR dataset path collection and row shaping helpers."""

from pathlib import Path

from src.runner.run_ocr_dataset import _build_record, _collect_images, _infer_condition


def test_infer_condition_uses_parent_directory():
    """Condition names should come from the first directory below dataset root."""
    root = Path("data/ocr_dataset")
    image = root / "dim" / "sample.png"
    assert _infer_condition(image, root) == "dim"


def test_collect_images_respects_patterns(tmp_path):
    """Image collection should include only files matching the configured glob patterns."""
    (tmp_path / "a.png").write_text("x", encoding="utf-8")
    (tmp_path / "b.jpg").write_text("x", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")

    paths = _collect_images(tmp_path, recursive=False, patterns=["*.png", "*.jpg"])
    assert [path.name for path in paths] == ["a.png", "b.jpg"]


def test_build_record_preserves_requested_ocr_fields(tmp_path):
    """OCR row records should expose the fields needed for comparison tables."""
    record = _build_record(
        engine="tesseract",
        image_path=tmp_path / "sample.png",
        condition="bright_clean",
        runtime_ms=12.5,
        result={
            "ok": True,
            "error": None,
            "outputs": {"text": "hello", "confidence": 0.92},
        },
    )

    assert record["engine"] == "tesseract"
    assert record["image_id"] == "sample"
    assert record["condition"] == "bright_clean"
    assert record["runtime_ms"] == 12.5
    assert record["raw_text"] == "hello"
    assert record["confidence"] == 0.92
    assert record["human_verdict"] is None
