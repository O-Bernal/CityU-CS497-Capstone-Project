# CityU CS497 Capstone Project

Reduced-scope computer vision comparison pipeline for a capstone study using live webcam runs.

## Scope
- Face detection comparison
  - OpenCV Haar face detector
  - MediaPipe face detector
- Object recognition comparison
  - OpenCV DNN object detector
  - MediaPipe object detector
- OCR comparison
  - Tesseract
  - EasyOCR

## Repo Layout
- `src/core`: shared camera, config, logging, metrics, and CSV reporting helpers
- `src/tasks`: task-library adapters for face detection, object recognition, and OCR
- `src/runner`: live comparison runner, single-task runner, and export runner
- `configs`: task and comparison configs
- `models`: local detector model files for MediaPipe and OpenCV
- `data/logs`: raw JSON logs from live and dataset runs
- `results/summaries`: timestamped comparison JSON/CSV outputs
- `results/tables`: exported analysis-ready summary tables

## Current Workflow
Run commands from the repo root with the project interpreter:

```powershell
.\.venv\Scripts\python.exe
```

## Required Models
- Face detection, MediaPipe:
  - `models/mediapipe/blaze_face_short_range.tflite`
- Object recognition, MediaPipe:
  - `models/mediapipe/efficientdet_lite0.tflite`
- Object recognition, OpenCV DNN:
  - `models/opencv/frozen_inference_graph.pb`
  - `models/opencv/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt`

If your filenames differ, update the corresponding YAML config paths.

## Main Commands
- Face comparison:
```powershell
.\.venv\Scripts\python.exe -m src.runner.run_comparison --config configs/face_comparison.yaml
```

- Object comparison:
```powershell
.\.venv\Scripts\python.exe -m src.runner.run_comparison --config configs/object_comparison.yaml
```

- OCR comparison:
```powershell
.\.venv\Scripts\python.exe -m src.runner.run_comparison --config configs/ocr_comparison_live.yaml
```

- Single live task run:
```powershell
.\.venv\Scripts\python.exe -m src.runner.run_single_task --config configs/task_human.yaml
.\.venv\Scripts\python.exe -m src.runner.run_single_task --config configs/task_object.yaml
.\.venv\Scripts\python.exe -m src.runner.run_single_task --config configs/task_ocr_live.yaml
```

- Single live task run with presets:
```powershell
.\.venv\Scripts\python.exe -m src.runner.run_single_task --preset human
.\.venv\Scripts\python.exe -m src.runner.run_single_task --preset object
.\.venv\Scripts\python.exe -m src.runner.run_single_task --preset ocr
```

- Export CSV tables from collected logs:
```powershell
.\.venv\Scripts\python.exe -m src.runner.export_results
```

## Notes
- `configs/task_ocr_live.yaml` runs one OCR engine at a time using `task.library`.
- `configs/ocr_comparison_live.yaml` is the correct config for Tesseract vs EasyOCR comparison.
- Object comparison conditions are tuned for detector-friendly classes:
  - `bottle`
  - `cup`
  - `book`
  - `cell phone`
- `results/tables` is the main output for report analysis. Figures can be generated later outside the repo from the exported CSVs.
