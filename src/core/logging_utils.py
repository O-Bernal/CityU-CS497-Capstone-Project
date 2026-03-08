"""Utilities for writing structured JSON logs for repeatable experiments."""

from datetime import datetime
import json
from pathlib import Path


def safe_name(text: str) -> str:
    """Convert arbitrary text into a filesystem-safe slug."""
    out = []
    for ch in str(text).strip().lower():
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "run"


def timestamp_string() -> str:
    """Return a consistent timestamp string for generated artifacts."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_run_log(payload: dict, out_dir: str = "data/logs", stem: str = "run") -> Path:
    """Write one JSON payload to a timestamped log file and return the path."""
    ts = timestamp_string()
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    out_file = path / f"{safe_name(stem)}_{ts}.json"
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_file
