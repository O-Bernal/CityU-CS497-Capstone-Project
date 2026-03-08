# CityU CS497 Capstone Project

Task-focused comparison pipeline for a capstone study on webcam face detection and OCR.

## Scope
- Face detection comparison
  - OpenCV Haar face/person baseline
  - MediaPipe face detection
- OCR comparison
  - Tesseract
  - EasyOCR

## Repo Layout
- `src/core`: shared camera, config, logging, metrics, and CSV reporting helpers
- `src/tasks`: task-library adapters
- `src/runner`: live face comparison and OCR dataset runners
- `configs`: focused run configs for face detection and OCR
- `data/logs`: raw JSON run logs
- `results/summaries`: timestamped CSV/JSON outputs for experiments
- `results/tables`: regenerated summary tables for paper use

## Quick Start
1. Create a virtual environment.
2. Install dependencies from `requirements.txt`.
3. Install the Tesseract executable separately. On Windows, a typical path is `C:\Program Files\Tesseract-OCR\tesseract.exe`.
4. If Tesseract is not on `PATH`, set `ocr.tesseract_cmd` in `configs/task_ocr.yaml` or set the `TESSERACT_CMD` environment variable.
5. Place OCR dataset images under `data/ocr_dataset`, preferably grouped by condition folder.
6. Run `python -m src.runner.run_comparison --config configs/face_comparison.yaml`.
7. Run `python -m src.runner.run_ocr_dataset --config configs/task_ocr.yaml`.
8. Run `python scripts/export_results.py`.
