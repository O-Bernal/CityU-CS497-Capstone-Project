"""Convert raw JSON result logs into summary CSV tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.reporting import write_csv_rows


def _load_payload(path: Path) -> dict:
    """Read a JSON payload from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_face_rows(payload: dict, path: Path) -> list[dict]:
    """Extract face-detection rows from a live task payload."""
    record = payload.get("record")
    if not isinstance(record, dict):
        return []
    if record.get("task") != "human_cues":
        return []

    row = dict(record)
    row["log_path"] = str(path)
    return [row]


def _collect_ocr_rows(payload: dict, path: Path) -> list[dict]:
    """Extract OCR rows from a dataset payload."""
    if payload.get("task") != "ocr":
        return []

    rows = []
    for record in payload.get("records", []):
        if not isinstance(record, dict):
            continue
        row = dict(record)
        row["log_path"] = str(path)
        rows.append(row)
    return rows


def main() -> None:
    """Export face-detection and OCR summary tables from saved JSON logs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-root", default="data/logs")
    parser.add_argument("--output-dir", default="results/tables")
    args = parser.parse_args()

    logs_root = Path(args.logs_root)
    output_dir = Path(args.output_dir)

    face_rows = []
    ocr_rows = []
    for path in sorted(logs_root.rglob("*.json")):
        payload = _load_payload(path)
        face_rows.extend(_collect_face_rows(payload, path))
        ocr_rows.extend(_collect_ocr_rows(payload, path))

    face_rows.sort(key=lambda row: (row.get("condition", ""), row.get("library", ""), row.get("repeat", 0)))
    ocr_rows.sort(key=lambda row: (row.get("condition", ""), row.get("image_id", ""), row.get("engine", "")))

    output_dir.mkdir(parents=True, exist_ok=True)
    face_path = write_csv_rows(face_rows, output_dir / "face_detection_summary.csv")
    ocr_path = write_csv_rows(ocr_rows, output_dir / "ocr_summary.csv")

    print(f"Wrote {face_path}")
    print(f"Wrote {ocr_path}")


if __name__ == "__main__":
    main()
