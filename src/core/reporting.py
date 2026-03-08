"""Helpers for exporting experiment records into paper-ready CSV tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _stringify(value: Any) -> str | int | float:
    """Convert nested values into CSV-safe scalar representations."""
    if value is None:
        return ""
    if isinstance(value, (str, int, float)):
        return value
    return json.dumps(value, sort_keys=True)


def write_csv_rows(rows: list[dict[str, Any]], out_path: str | Path) -> Path:
    """Write row dictionaries to a CSV file, preserving first-seen column order."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return path

    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _stringify(row.get(key)) for key in fieldnames})

    return path
