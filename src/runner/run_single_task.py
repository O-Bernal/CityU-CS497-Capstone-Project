"""Run one configured webcam task and write a structured metrics log."""

import argparse
from collections import Counter
from datetime import datetime
from importlib import import_module
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.camera import Camera
from src.core.config import load_config
from src.core.logging_utils import safe_name, write_run_log
from src.core.metrics import RunMetrics
from src.runner.task_selection import select_library
from src.tasks.interface import TaskResult
from src.tasks.registry import get_task_runner


TASK_PRESET_CONFIGS = {
    "human": "configs/task_human.yaml",
    "object": "configs/task_object.yaml",
    "ocr": "configs/task_ocr_live.yaml",
}


def _create_video_writer(path: Path, fps: float, size: tuple[int, int]):
    """Create a video writer for webcam capture output."""
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc_factory = getattr(cv2, "VideoWriter_fourcc", None)
    if fourcc_factory is None:
        fourcc_factory = getattr(cv2.VideoWriter, "fourcc", None)
    if fourcc_factory is None:
        raise RuntimeError("OpenCV build does not expose a VideoWriter fourcc helper.")

    fourcc = fourcc_factory(*"mp4v")
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


def _warm_up_camera(camera: Camera, warmup_frames: int) -> int:
    """Discard a fixed number of frames before collecting timed metrics."""
    completed = 0
    while completed < warmup_frames:
        ok, _frame = camera.read()
        if ok:
            completed += 1
    return completed


def _build_log_stem(task_name: str, library_name: str, condition: str, repeat: int) -> str:
    """Build a readable log filename stem for one experiment run."""
    return "_".join(
        [
            safe_name(task_name),
            safe_name(library_name),
            safe_name(condition),
            f"r{repeat}",
        ]
    )


def _build_run_record(
    *,
    task_name: str,
    library_name: str,
    condition: str,
    repeat: int,
    summary: dict,
    open_ms: float,
    warmup_frames: int,
    frames_with_detection: int,
    label_counts: Counter[str],
    failed_frames: int,
    avg_confidence: float | None,
    resolution: tuple[int | None, int | None],
    video_path: Path | None,
    output_text: str | None,
    matched_label: str | None,
) -> dict:
    """Create a flat record used by JSON logs and CSV summary exports."""
    frames_processed = int(summary.get("frame_count", 0))
    duration_s = float(summary.get("duration_s", 0.0))
    return {
        "task": task_name,
        "library": library_name,
        "condition": condition,
        "repeat": repeat,
        "frames_processed": frames_processed,
        "failed_frames": failed_frames,
        "duration_s": duration_s,
        "fps": float(summary.get("fps", 0.0)),
        "avg_processing_ms": float(summary.get("avg_processing_ms", 0.0)),
        "webcam_open_ms": open_ms,
        "warmup_frames": warmup_frames,
        "frame_width": resolution[0],
        "frame_height": resolution[1],
        "frames_with_detection": frames_with_detection,
        "detection_rate": (frames_with_detection / frames_processed) if frames_processed else 0.0,
        "label_counts": dict(label_counts),
        "avg_confidence": avg_confidence,
        "output_text": output_text,
        "matched_label": matched_label,
        "video_path": str(video_path) if video_path is not None else None,
        "verdict": None,
        "notes": None,
    }


def run_task(cfg: dict, *, write_log: bool = True) -> tuple[dict, str | None]:
    """Execute one task run from an in-memory config and return payload/log path."""
    task_cfg = cfg.get("task", {})
    task_name = str(task_cfg.get("name", "")).strip()
    if not task_name:
        raise ValueError("Task config must define 'task.name'.")

    library_name = select_library(task_cfg)
    task_runner = get_task_runner(task_name, library_name)
    task_module = import_module(task_runner.__module__)
    configure_task = getattr(task_module, "configure", None)
    if callable(configure_task):
        configure_task(cfg)

    run_cfg = cfg.get("run", {})
    max_frames = int(run_cfg.get("max_frames", 120))
    max_seconds = run_cfg.get("max_seconds")
    max_seconds = float(max_seconds) if max_seconds is not None else None
    warmup_frames = int(run_cfg.get("warmup_frames", 0))
    record_video = bool(run_cfg.get("record_video", False))
    show_preview = bool(run_cfg.get("show_preview", False))
    video_dir = Path(run_cfg.get("video_dir", "data/captures"))
    log_dir = str(run_cfg.get("log_dir", f"data/logs/{safe_name(task_name)}"))

    experiment = cfg.get("experiment", {})
    condition = str(experiment.get("condition", "default"))
    repeat = int(experiment.get("repeat", 1))

    camera_cfg = cfg.get("camera", {})
    camera = Camera(
        index=int(camera_cfg.get("index", 0)),
        width=camera_cfg.get("width"),
        height=camera_cfg.get("height"),
    )

    failed_frames = 0
    last_result: TaskResult | None = None
    label_counts: Counter[str] = Counter()
    confidence_values: list[float] = []
    frames_with_detection = 0
    observed_width = None
    observed_height = None
    last_output_text = None
    last_matched_label = None

    print(f"[INFO] Opening webcam index {camera.index}...")
    open_start = time.perf_counter()
    camera.open()
    open_ms = (time.perf_counter() - open_start) * 1000.0
    print(f"[INFO] Webcam opened in {open_ms:.1f} ms")

    if warmup_frames > 0:
        print(f"[INFO] Warming up camera for {warmup_frames} frames...")
        _warm_up_camera(camera, warmup_frames)

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

            frame_idx += 1
            observed_height, observed_width = frame.shape[:2]

            if record_video and writer is None:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                stem = "_".join(
                    [
                        safe_name(task_name),
                        safe_name(library_name),
                        safe_name(condition),
                        f"r{repeat}",
                        ts,
                    ]
                )
                video_path = video_dir / f"{stem}.mp4"
                writer = _create_video_writer(video_path, fps=20.0, size=(observed_width, observed_height))

            start = time.perf_counter()
            try:
                last_result = task_runner(frame)
            except Exception as exc:  # noqa: BLE001
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
                if isinstance(outputs, dict):
                    raw_text = outputs.get("text")
                    if raw_text:
                        cleaned_text = str(raw_text).strip()
                        if cleaned_text:
                            last_output_text = cleaned_text
                    raw_label = outputs.get("matched_label")
                    if raw_label:
                        cleaned_label = str(raw_label).strip()
                        if cleaned_label:
                            last_matched_label = cleaned_label
            else:
                failed_frames += 1

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
                if last_output_text:
                    cv2.putText(
                        preview,
                        last_output_text[:80],
                        (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 220, 255),
                        2,
                    )
                elif last_matched_label:
                    cv2.putText(
                        preview,
                        f"match {last_matched_label}",
                        (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 220, 255),
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
    avg_confidence = (sum(confidence_values) / len(confidence_values)) if confidence_values else None
    reported_resolution = (
        observed_width,
        observed_height,
    ) if observed_width is not None and observed_height is not None else camera.actual_resolution()
    record = _build_run_record(
        task_name=task_name,
        library_name=library_name,
        condition=condition,
        repeat=repeat,
        summary=summary,
        open_ms=open_ms,
        warmup_frames=warmup_frames,
        frames_with_detection=frames_with_detection,
        label_counts=label_counts,
        failed_frames=failed_frames,
        avg_confidence=avg_confidence,
        resolution=reported_resolution,
        video_path=video_path,
        output_text=last_output_text,
        matched_label=last_matched_label,
    )

    run_payload = {
        "schema_version": 1,
        "result_type": "live_task_run",
        "record": record,
        "config": cfg,
        "experiment": {
            "condition": condition,
            "repeat": repeat,
        },
        "execution": {
            "task": task_name,
            "library": library_name,
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
            "frames_with_detection": frames_with_detection,
            "detection_rate": record["detection_rate"],
            "label_counts": dict(label_counts),
            "avg_confidence": avg_confidence,
        },
        "artifacts": {
            "video_path": str(video_path) if video_path is not None else None,
        },
        "review": {
            "verdict": None,
            "notes": None,
        },
        "summary": summary,
    }

    out_file = None
    if write_log:
        out_file = str(
            write_run_log(
                run_payload,
                out_dir=log_dir,
                stem=_build_log_stem(task_name, library_name, condition, repeat),
            )
        )
        print(f"Run complete. task={task_name}, library={library_name}, log={out_file}")

    return run_payload, out_file


def main() -> None:
    """Execute a single task run using the provided YAML configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--preset", choices=sorted(TASK_PRESET_CONFIGS))
    args = parser.parse_args()

    config_path = args.config or (TASK_PRESET_CONFIGS.get(args.preset) if args.preset else None)
    if not config_path:
        raise ValueError("Provide --config <path> or --preset human|object|ocr.")

    cfg = load_config(config_path)
    run_task(cfg, write_log=True)


if __name__ == "__main__":
    main()
