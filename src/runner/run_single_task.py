"""Run one configured webcam task and write a metrics log for that session."""

import argparse
import time

from src.core.camera import Camera
from src.core.config import load_config
from src.core.metrics import RunMetrics
from src.core.logging_utils import write_run_log


def main() -> None:
    """Execute a single task run using the provided YAML configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    max_frames = int(cfg.get("run", {}).get("max_frames", 120))

    camera = Camera(index=int(cfg.get("camera", {}).get("index", 0)))
    metrics = RunMetrics()

    camera.open()
    try:
        for _ in range(max_frames):
            start = time.perf_counter()
            ok, _frame = camera.read()
            if not ok:
                break
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            metrics.record_frame(elapsed_ms)
    finally:
        camera.close()

    summary = metrics.summary()
    out_file = write_run_log({"config": cfg, "summary": summary})
    print(f"Run complete. Log: {out_file}")


if __name__ == "__main__":
    main()
