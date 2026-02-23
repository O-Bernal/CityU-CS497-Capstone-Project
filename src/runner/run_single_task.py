"""Run one configured webcam task and write a metrics log for that session."""

import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
import time

from src.core.camera import Camera
from src.core.config import load_config
from src.core.logging_utils import write_run_log
from src.core.metrics import RunMetrics
from src.runner.task_selection import select_library
from src.tasks.interface import TaskResult
from src.tasks.registry import get_task_runner


def _safe_name(text: str) -> str:
    """Return a filesystem-safe name segment for generated artifacts."""
    out = []
    for ch in text.strip().lower():
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "run"


def _create_video_writer(path: Path, fps: float, size: tuple[int, int]):
    """Create a video writer for webcam capture output."""
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if writer.isOpened():
        return writer

    writer.release()
    raise RuntimeError(f"Unable to open video writer for {path}")


def _draw_detections(preview, detections: list[dict], cv2_module) -> None:
    """Render standardized detections on preview frames."""
    for det in detections:
        bbox = det.get("bbox") or []
        if len(bbox) != 4:
            continue
        x, y, w, h = [int(v) for v in bbox]
        label = str(det.get("label", "object"))
        conf = det.get("confidence")
        caption = f"{label}"
        if conf is not None:
            caption += f" {float(conf):.2f}"

        cv2_module.rectangle(preview, (x, y), (x + w, y + h), (30, 200, 30), 2)
        cv2_module.putText(
            preview,
            caption,
            (x, max(20, y - 8)),
            cv2_module.FONT_HERSHEY_SIMPLEX,
            0.5,
            (30, 200, 30),
            2,
        )


def run_task(cfg: dict, *, write_log: bool = True) -> tuple[dict, str | None]:
    """Execute one task run from an in-memory config and return payload/log path."""
    run_cfg = cfg.get("run", {})
    max_frames = int(run_cfg.get("max_frames", 120))
    max_seconds = run_cfg.get("max_seconds")
    max_seconds = float(max_seconds) if max_seconds is not None else None
    record_video = bool(run_cfg.get("record_video", False))
    show_preview = bool(run_cfg.get("show_preview", False))
    video_dir = Path(run_cfg.get("video_dir", "data/captures"))

    task_cfg = cfg.get("task", {})
    task_name = str(task_cfg.get("name", "")).strip()
    if not task_name:
        raise ValueError("Task config must define 'task.name'.")

    library_name = select_library(task_cfg)
    task_runner = get_task_runner(task_name, library_name)

    experiment = cfg.get("experiment", {})
    condition = str(experiment.get("condition", "default"))
    repeat = int(experiment.get("repeat", 1))

    camera = Camera(index=int(cfg.get("camera", {}).get("index", 0)))

    processed_frames = 0
    failed_frames = 0
    last_result: TaskResult | None = None
    label_counts: Counter[str] = Counter()
    confidence_values: list[float] = []
    frames_with_detection = 0

    print(f"[INFO] Opening webcam index {camera.index}...")
    open_start = time.perf_counter()
    camera.open()
    open_ms = (time.perf_counter() - open_start) * 1000.0
    print(f"[INFO] Webcam opened in {open_ms:.1f} ms")

    metrics = RunMetrics()
    capture_started_at = time.perf_counter()

    writer = None
    video_path = None

    cv2 = None
    if show_preview:
        import cv2 as _cv2

        cv2 = _cv2

    try:
        frame_idx = 0
        while True:
            if frame_idx >= max_frames:
                break
            if max_seconds is not None and (time.perf_counter() - capture_started_at) >= max_seconds:
                break

            ok, frame = camera.read()
            if not ok:
                failed_frames += 1
                continue

            if record_video and writer is None:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                stem = "_".join(
                    [
                        _safe_name(task_name),
                        _safe_name(library_name),
                        _safe_name(condition),
                        f"r{repeat}",
                        ts,
                    ]
                )
                video_path = video_dir / f"{stem}.mp4"
                height, width = frame.shape[:2]
                writer = _create_video_writer(video_path, fps=20.0, size=(width, height))

            frame_idx += 1
            start = time.perf_counter()
            try:
                last_result = task_runner(frame)
                if not last_result.get("ok", False):
                    failed_frames += 1
                else:
                    processed_frames += 1
            except Exception as exc:  # noqa: BLE001
                failed_frames += 1
                last_result = {
                    "task": task_name,
                    "library": library_name,
                    "ok": False,
                    "outputs": {},
                    "error": str(exc),
                }

            elapsed_ms = (time.perf_counter() - start) * 1000.0
            metrics.record_frame(elapsed_ms)

            detections = []
            if last_result and last_result.get("ok", False):
                outputs = last_result.get("outputs", {})
                detections = outputs.get("detections", []) if isinstance(outputs, dict) else []

            if detections:
                frames_with_detection += 1
                for det in detections:
                    label = str(det.get("label", "unknown"))
                    label_counts[label] += 1
                    conf = det.get("confidence")
                    if conf is not None:
                        confidence_values.append(float(conf))

            if writer is not None:
                writer.write(frame)

            if show_preview and cv2 is not None:
                preview = frame.copy()
                if detections:
                    _draw_detections(preview, detections, cv2)
                cv2.putText(
                    preview,
                    f"{task_name}/{library_name} | {condition} | r{repeat}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    preview,
                    f"frame {frame_idx} | q=stop",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("Run Preview", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] Preview quit requested; ending run early.")
                    break
    finally:
        if writer is not None:
            writer.release()
        camera.close()
        if show_preview and cv2 is not None:
            cv2.destroyAllWindows()

    summary = metrics.summary()
    detection_summary = {
        "frames_with_detection": frames_with_detection,
        "detection_rate": (frames_with_detection / max(summary.get("frame_count", 1), 1)),
        "label_counts": dict(label_counts),
        "avg_confidence": (
            (sum(confidence_values) / len(confidence_values)) if confidence_values else None
        ),
    }

    run_payload = {
        "config": cfg,
        "experiment": {
            "condition": condition,
            "repeat": repeat,
        },
        "execution": {
            "task": task_name,
            "library": library_name,
            "processed_frames": processed_frames,
            "failed_frames": failed_frames,
            "last_result": last_result,
        },
        "timing": {
            "open_ms": open_ms,
            "run_limit": {
                "max_frames": max_frames,
                "max_seconds": max_seconds,
            },
        },
        "metrics": {
            "detection_summary": detection_summary,
        },
        "artifacts": {
            "video_path": str(video_path) if video_path is not None else None,
        },
        "summary": summary,
    }

    out_file = None
    if write_log:
        out_file = str(write_run_log(run_payload))
        print(f"Run complete. task={task_name}, library={library_name}, log={out_file}")

    return run_payload, out_file


def main() -> None:
    """Execute a single task run using the provided YAML configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_task(cfg, write_log=True)


if __name__ == "__main__":
    main()
