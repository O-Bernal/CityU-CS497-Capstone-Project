"""Execute a configured comparison matrix across tasks, libraries, and conditions."""

import argparse
import copy
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.config import load_config
from src.core.logging_utils import safe_name, timestamp_string, write_run_log
from src.core.reporting import write_csv_rows
from src.runner.export_results import export_logs
from src.runner.run_single_task import run_task
from src.tasks.registry import TASK_MODULES


def _expand_matrix(comp_cfg: dict) -> list[tuple[str, str, str, int]]:
    """Build the set of (task, library, condition, repeat) experiments to run."""
    default_conditions = comp_cfg.get("conditions", ["default"]) or ["default"]
    repeats = int(comp_cfg.get("repeats", 1))
    task_conditions = comp_cfg.get("task_conditions", {})

    task_libraries = comp_cfg.get("task_libraries")
    if isinstance(task_libraries, dict) and task_libraries:
        tasks = list(task_libraries.keys())
    else:
        tasks = comp_cfg.get("tasks", [])

    all_pairs = []
    for task in tasks:
        if isinstance(task_libraries, dict) and task in task_libraries:
            libs = task_libraries[task]
        else:
            libs = comp_cfg.get("libraries", [])

        conditions = task_conditions.get(task, default_conditions) if isinstance(task_conditions, dict) else default_conditions
        conditions = conditions or ["default"]

        for condition in conditions:
            for lib in libs:
                if (task, lib) not in TASK_MODULES:
                    print(f"[WARN] Skipping unsupported pair: {task}/{lib}")
                    continue
                for repeat in range(1, repeats + 1):
                    all_pairs.append((str(task), str(lib), str(condition), repeat))

    return all_pairs


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


def _show_condition_preview(camera_index: int, condition: str, width: int | None, height: int | None) -> bool:
    """Show a live setup preview and wait for SPACE to start condition runs."""
    import cv2

    from src.core.camera import Camera

    cam = Camera(index=camera_index, width=width, height=height)
    cam.open()
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                continue

            cv2.putText(
                frame,
                f"Setup condition: {condition}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                "SPACE=start, ESC=cancel",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Condition Setup Preview", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # SPACE
                return True
            if key == 27:  # ESC
                return False
    finally:
        cam.close()
        cv2.destroyAllWindows()


def _wait_for_condition_ready(
    *,
    condition: str,
    camera_index: int,
    width: int | None,
    height: int | None,
    interactive: bool,
    preview: bool,
) -> bool:
    """Gate each condition so the user can physically set the environment first."""
    if not interactive:
        return True

    print()
    print(f"[SETUP] Condition: {condition}")
    print("[SETUP] Set the scene now (lighting, motion, background, distance).")

    if preview:
        print("[SETUP] Opening preview. Press SPACE to start this condition, ESC to cancel.")
        return _show_condition_preview(camera_index, condition, width, height)

    answer = input("[SETUP] Press Enter to start, or type 'q' to stop: ").strip().lower()
    return answer != "q"


def _wait_for_run_ready(*, enabled: bool, task: str, library: str, condition: str, repeat: int) -> bool:
    """Optionally gate each run so the operator can start each repeat deliberately."""
    if not enabled:
        return True

    print(
        f"[RUN] Ready: task={task}, library={library}, condition={condition}, repeat={repeat}. "
        "Press Enter to start or type 'q' to stop."
    )
    answer = input("[RUN] Start run: ").strip().lower()
    return answer != "q"


def main() -> None:
    """Run all configured comparison experiments and write summary artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    comp_cfg = base_cfg.get("comparison", {})

    experiments = _expand_matrix(comp_cfg)
    if not experiments:
        raise ValueError("No valid comparison experiments generated from config.")

    interactive_conditions = bool(comp_cfg.get("interactive_conditions", True))
    condition_preview = bool(comp_cfg.get("condition_preview", True))
    pause_before_run = bool(comp_cfg.get("pause_before_run", False))
    camera_cfg = base_cfg.get("camera", {})
    camera_index = int(camera_cfg.get("index", 0))
    camera_width = int(camera_cfg["width"]) if camera_cfg.get("width") is not None else None
    camera_height = int(camera_cfg["height"]) if camera_cfg.get("height") is not None else None
    comparison_name = str(comp_cfg.get("name", "comparison"))
    log_dir = str(comp_cfg.get("log_dir", "data/logs/comparison"))
    summary_dir = Path(comp_cfg.get("summary_dir", "results/summaries"))
    task_run_overrides = comp_cfg.get("task_run_overrides", {})
    export_after_run = bool(comp_cfg.get("export_after_run", True))
    export_logs_root = comp_cfg.get("export_logs_root", "data/logs")
    export_output_dir = comp_cfg.get("export_output_dir", "results/tables")

    total = len(experiments)
    print(f"[INFO] Running {total} comparison experiments...")

    run_summaries = []
    active_condition = None
    for idx, (task, library, condition, repeat) in enumerate(experiments, start=1):
        if condition != active_condition:
            ready = _wait_for_condition_ready(
                condition=condition,
                camera_index=camera_index,
                width=camera_width,
                height=camera_height,
                interactive=interactive_conditions,
                preview=condition_preview,
            )
            if not ready:
                print("[INFO] Comparison cancelled during condition setup.")
                break
            active_condition = condition

        print(
            f"[INFO] ({idx}/{total}) task={task}, library={library}, condition={condition}, repeat={repeat}"
        )
        if not _wait_for_run_ready(
            enabled=pause_before_run,
            task=task,
            library=library,
            condition=condition,
            repeat=repeat,
        ):
            print("[INFO] Comparison stopped before run start.")
            break

        cfg = copy.deepcopy(base_cfg)
        cfg["task"] = {
            "name": task,
            "library": library,
            "libraries": [library],
        }

        cfg.setdefault("run", {})
        if "max_frames" in comp_cfg:
            cfg["run"]["max_frames"] = comp_cfg["max_frames"]
        if "max_seconds" in comp_cfg:
            cfg["run"]["max_seconds"] = comp_cfg["max_seconds"]
        if "warmup_frames" in comp_cfg:
            cfg["run"]["warmup_frames"] = comp_cfg["warmup_frames"]
        if "record_video" in comp_cfg:
            cfg["run"]["record_video"] = bool(comp_cfg["record_video"])
        if "video_dir" in comp_cfg:
            cfg["run"]["video_dir"] = comp_cfg["video_dir"]
        if "show_preview" in comp_cfg:
            cfg["run"]["show_preview"] = bool(comp_cfg["show_preview"])
        cfg["run"]["log_dir"] = log_dir

        if isinstance(task_run_overrides, dict):
            overrides = task_run_overrides.get(task, {})
            if isinstance(overrides, dict):
                cfg["run"].update(overrides)

        cfg["experiment"] = {
            "condition": condition,
            "repeat": repeat,
        }

        payload, _unused_log_path = run_task(cfg, write_log=False)
        review = {"verdict": None, "notes": None}
        payload["review"] = review
        record = dict(payload.get("record", {}))
        record["verdict"] = review.get("verdict")
        record["notes"] = review.get("notes")
        payload["record"] = record

        log_path = write_run_log(
            payload,
            out_dir=log_dir,
            stem=_build_log_stem(task, library, condition, repeat),
        )

        run_row = dict(record)
        run_row["log_path"] = str(log_path)
        run_summaries.append(run_row)

    summary_dir.mkdir(parents=True, exist_ok=True)
    ts = timestamp_string()
    summary_stem = safe_name(comparison_name)

    json_path = summary_dir / f"{summary_stem}_{ts}.json"
    csv_path = summary_dir / f"{summary_stem}_{ts}.csv"

    json_path.write_text(
        json.dumps(
            {
                "source_config": args.config,
                "comparison_name": comparison_name,
                "generated_at": ts,
                "count": len(run_summaries),
                "runs": run_summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_csv_rows(run_summaries, csv_path)

    print(f"[INFO] Comparison complete. Summary JSON: {json_path}")
    print(f"[INFO] Comparison complete. Summary CSV: {csv_path}")

    if export_after_run:
        outputs = export_logs(logs_root=export_logs_root, output_dir=export_output_dir)
        print(f"[INFO] Updated cumulative table: {outputs['face']}")
        print(f"[INFO] Updated cumulative table: {outputs['object']}")
        print(f"[INFO] Updated cumulative table: {outputs['ocr']}")


if __name__ == "__main__":
    main()
