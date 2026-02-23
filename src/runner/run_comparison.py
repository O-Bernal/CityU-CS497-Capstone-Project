"""Execute a configured comparison matrix across tasks, libraries, and conditions."""

import argparse
import copy
from datetime import datetime
import json
from pathlib import Path

from src.core.config import load_config
from src.runner.run_single_task import run_task
from src.tasks.registry import TASK_MODULES


def _expand_matrix(comp_cfg: dict) -> list[tuple[str, str, str, int]]:
    """Build the set of (task, library, condition, repeat) experiments to run."""
    conditions = comp_cfg.get("conditions", ["default"]) or ["default"]
    repeats = int(comp_cfg.get("repeats", 1))

    task_libraries = comp_cfg.get("task_libraries")
    if isinstance(task_libraries, dict) and task_libraries:
        tasks = list(task_libraries.keys())
    else:
        tasks = comp_cfg.get("tasks", [])

    all_pairs = []
    for condition in conditions:
        for task in tasks:
            if isinstance(task_libraries, dict) and task in task_libraries:
                libs = task_libraries[task]
            else:
                libs = comp_cfg.get("libraries", [])

            for lib in libs:
                if (task, lib) not in TASK_MODULES:
                    print(f"[WARN] Skipping unsupported pair: {task}/{lib}")
                    continue
                for repeat in range(1, repeats + 1):
                    all_pairs.append((str(task), str(lib), str(condition), repeat))

    return all_pairs


def _show_condition_preview(camera_index: int, condition: str) -> bool:
    """Show a live setup preview and wait for SPACE to start condition runs."""
    import cv2

    from src.core.camera import Camera

    cam = Camera(index=camera_index)
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
    *, condition: str, camera_index: int, interactive: bool, preview: bool
) -> bool:
    """Gate each condition so the user can physically set the environment first."""
    if not interactive:
        return True

    print()
    print(f"[SETUP] Condition: {condition}")
    print("[SETUP] Set the scene now (lighting, motion, background, distance).")

    if preview:
        print("[SETUP] Opening preview. Press SPACE to start this condition, ESC to cancel.")
        return _show_condition_preview(camera_index, condition)

    answer = input("[SETUP] Press Enter to start, or type 'q' to stop: ").strip().lower()
    return answer != "q"


def _prompt_run_verdict(*, enabled: bool) -> tuple[bool, dict]:
    """Collect reviewer verdict after each run and optionally stop the campaign."""
    if not enabled:
        return True, {"verdict": None, "notes": None}

    print("[REVIEW] Enter verdict: [c]orrect, [i]ncorrect, [u]ncertain, [q]uit")
    while True:
        verdict_raw = input("[REVIEW] Verdict: ").strip().lower()
        if verdict_raw in {"c", "correct"}:
            verdict = "correct"
            break
        if verdict_raw in {"i", "incorrect"}:
            verdict = "incorrect"
            break
        if verdict_raw in {"u", "uncertain"}:
            verdict = "uncertain"
            break
        if verdict_raw in {"q", "quit"}:
            return False, {"verdict": "quit", "notes": None}
        print("[REVIEW] Invalid input. Use c / i / u / q.")

    notes = input("[REVIEW] Optional notes (Enter to skip): ").strip()
    return True, {"verdict": verdict, "notes": notes or None}


def main() -> None:
    """Run all configured comparison experiments and write a summary artifact."""
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
    pause_after_run = bool(comp_cfg.get("pause_after_run", True))
    camera_index = int(base_cfg.get("camera", {}).get("index", 0))

    total = len(experiments)
    print(f"[INFO] Running {total} comparison experiments...")

    run_summaries = []
    active_condition = None
    for idx, (task, library, condition, repeat) in enumerate(experiments, start=1):
        if condition != active_condition:
            ready = _wait_for_condition_ready(
                condition=condition,
                camera_index=camera_index,
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
        if "record_video" in comp_cfg:
            cfg["run"]["record_video"] = bool(comp_cfg["record_video"])
        if "video_dir" in comp_cfg:
            cfg["run"]["video_dir"] = comp_cfg["video_dir"]
        if "show_preview" in comp_cfg:
            cfg["run"]["show_preview"] = bool(comp_cfg["show_preview"])

        cfg["experiment"] = {
            "condition": condition,
            "repeat": repeat,
        }

        payload, log_path = run_task(cfg, write_log=True)
        should_continue, review = _prompt_run_verdict(enabled=pause_after_run)

        run_summaries.append(
            {
                "task": task,
                "library": library,
                "condition": condition,
                "repeat": repeat,
                "log_path": log_path,
                "summary": payload.get("summary", {}),
                "timing": payload.get("timing", {}),
                "metrics": payload.get("metrics", {}),
                "artifacts": payload.get("artifacts", {}),
                "execution": payload.get("execution", {}),
                "review": review,
            }
        )

        if not should_continue:
            print("[INFO] Comparison stopped by user during run review.")
            break

    out_dir = Path("results/summaries")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"comparison_{ts}.json"

    out_payload = {
        "source_config": args.config,
        "generated_at": ts,
        "count": len(run_summaries),
        "runs": run_summaries,
    }
    out_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")

    print(f"[INFO] Comparison complete. Summary: {out_path}")


if __name__ == "__main__":
    main()
