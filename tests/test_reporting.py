"""Tests for CSV reporting helpers and raw-log export extraction."""

import json
from pathlib import Path

from src.core.reporting import write_csv_rows
from scripts.export_results import _collect_face_rows, _collect_ocr_rows


def test_write_csv_rows_stringifies_nested_values(tmp_path):
    """Nested dict/list values should be serialized safely for CSV output."""
    out_path = tmp_path / "rows.csv"
    write_csv_rows(
        [
            {
                "library": "opencv",
                "label_counts": {"face": 2, "person": 1},
                "notes": ["usable"],
            }
        ],
        out_path,
    )

    content = out_path.read_text(encoding="utf-8")
    assert "library" in content
    assert json.dumps({"face": 2, "person": 1}, sort_keys=True) in content


def test_collect_face_rows_reads_live_task_record():
    """Face export should pull the flat run record and attach its log path."""
    payload = {"record": {"task": "human_cues", "library": "opencv", "condition": "bright"}}
    log_path = Path("data/logs/test.json")
    rows = _collect_face_rows(payload, log_path)
    assert rows == [
        {
            "task": "human_cues",
            "library": "opencv",
            "condition": "bright",
            "log_path": str(log_path),
        }
    ]


def test_collect_ocr_rows_reads_dataset_records():
    """OCR export should expand every stored OCR record row."""
    payload = {
        "task": "ocr",
        "records": [
            {"engine": "tesseract", "image_id": "img01"},
            {"engine": "easyocr", "image_id": "img01"},
        ],
    }
    log_path = Path("data/logs/ocr.json")
    rows = _collect_ocr_rows(payload, log_path)
    assert len(rows) == 2
    assert rows[0]["log_path"] == str(log_path)
