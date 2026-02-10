"""Aggregate run logs into a single summary file for downstream analysis."""

from pathlib import Path
import json


def main() -> None:
    """Read all run logs and export a combined JSON summary artifact."""
    logs_dir = Path("data/logs")
    logs = sorted(logs_dir.glob("run_*.json"))
    payload = []
    for log in logs:
        payload.append(json.loads(log.read_text(encoding="utf-8")))
    out = Path("results/summaries/summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
