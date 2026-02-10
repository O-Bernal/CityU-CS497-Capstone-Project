"""Utilities for writing run artifacts to timestamped JSON logs."""

from pathlib import Path
import json
from datetime import datetime


def write_run_log(payload: dict, out_dir: str = "data/logs") -> Path:
    """Write one run payload to a timestamped JSON file and return its path."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    out_file = path / f"run_{ts}.json"
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_file
