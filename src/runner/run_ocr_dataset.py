"""Run OCR engines over a saved-image dataset and export structured results."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2

from src.core.config import load_config
from src.core.logging_utils import safe_name, timestamp_string, write_run_log
from src.core.reporting import write_csv_rows
from src.tasks.registry import get_task_runner


DEFAULT_PATTERNS = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]


def _collect_images(root: Path, *, recursive: bool, patterns: list[str]) -> list[Path]:
    """Collect dataset image files using the configured glob patterns."""
    image_paths: set[Path] = set()
    for pattern in patterns:
        matcher = root.rglob if recursive else root.glob
        image_paths.update(matcher(pattern))
    return sorted(path for path in image_paths if path.is_file())


def _infer_condition(image_path: Path, dataset_root: Path) -> str:
    """Infer a condition name from the first directory under the dataset root."""
    relative = image_path.relative_to(dataset_root)
    if len(relative.parts) > 1:
        return relative.parts[0]
    return "default"


def _build_record(
    *,
    engine: str,
    image_path: Path,
    condition: str,
    runtime_ms: float,
    result: dict,
) -> dict:
    """Build a flat OCR record row for JSON logs and CSV export."""
    outputs = result.get("outputs", {}) if result.get("ok", False) else {}
    return {
        "task": "ocr",
        "engine": engine,
        "image_id": image_path.stem,
        "condition": condition,
        "runtime_ms": runtime_ms,
        "raw_text": outputs.get("text", ""),
        "confidence": outputs.get("confidence"),
        "human_verdict": None,
        "ok": bool(result.get("ok", False)),
        "error": result.get("error"),
        "source_path": str(image_path),
    }


def main() -> None:
    """Run the configured OCR engines against every image in the dataset."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    task_cfg = cfg.get("task", {})
    dataset_cfg = cfg.get("dataset", {})
    ocr_cfg = cfg.get("ocr", {})

    tesseract_cmd = ocr_cfg.get("tesseract_cmd")
    if tesseract_cmd:
        os.environ["TESSERACT_CMD"] = str(tesseract_cmd)

    libraries = [str(lib) for lib in task_cfg.get("libraries", [])]
    explicit_library = task_cfg.get("library")
    if explicit_library and explicit_library not in libraries:
        libraries.insert(0, str(explicit_library))
    if not libraries:
        raise ValueError("OCR config must define task.library or task.libraries.")

    image_dir = dataset_cfg.get("image_dir")
    if not image_dir:
        raise ValueError("OCR config must define dataset.image_dir.")

    dataset_root = Path(image_dir)
    if not dataset_root.exists():
        raise FileNotFoundError(f"OCR dataset directory does not exist: {dataset_root}")

    recursive = bool(dataset_cfg.get("recursive", True))
    patterns = [str(pattern) for pattern in dataset_cfg.get("patterns", DEFAULT_PATTERNS)]
    images = _collect_images(dataset_root, recursive=recursive, patterns=patterns)
    if not images:
        raise ValueError(f"No dataset images found in {dataset_root}")

    log_dir = str(dataset_cfg.get("log_dir", "data/logs/ocr"))
    summary_dir = Path(dataset_cfg.get("summary_dir", "results/summaries"))
    run_name = str(dataset_cfg.get("name", "ocr_comparison"))

    records = []
    for engine in libraries:
        runner = get_task_runner("ocr", engine)
        print(f"[INFO] OCR engine={engine} images={len(images)}")

        for image_path in images:
            frame = cv2.imread(str(image_path))
            condition = _infer_condition(image_path, dataset_root)

            if frame is None:
                records.append(
                    {
                        "task": "ocr",
                        "engine": engine,
                        "image_id": image_path.stem,
                        "condition": condition,
                        "runtime_ms": 0.0,
                        "raw_text": "",
                        "confidence": None,
                        "human_verdict": None,
                        "ok": False,
                        "error": "Unable to read image file.",
                        "source_path": str(image_path),
                    }
                )
                continue

            start = time.perf_counter()
            result = runner(frame)
            runtime_ms = (time.perf_counter() - start) * 1000.0
            records.append(
                _build_record(
                    engine=engine,
                    image_path=image_path,
                    condition=condition,
                    runtime_ms=runtime_ms,
                    result=result,
                )
            )

    ts = timestamp_string()
    summary_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": 1,
        "result_type": "ocr_dataset_run",
        "task": "ocr",
        "dataset_root": str(dataset_root),
        "generated_at": ts,
        "count": len(records),
        "records": records,
    }

    log_path = write_run_log(payload, out_dir=log_dir, stem=run_name)
    csv_path = write_csv_rows(records, summary_dir / f"{safe_name(run_name)}_{ts}.csv")

    print(f"[INFO] OCR comparison complete. Raw log: {log_path}")
    print(f"[INFO] OCR comparison complete. Summary CSV: {csv_path}")


if __name__ == "__main__":
    main()
