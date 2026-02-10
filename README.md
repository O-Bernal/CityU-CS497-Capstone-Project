# CityU CS497 Capstone Project

Exploratory, comparative evaluation of webcam-based computer vision for interactive desktop use cases.

## Scope
- Object recognition
- OCR (text/character recognition)
- Human-centric cues (e.g., face/pose/gesture)

## Repo Layout
- `src/core`: shared camera, config, metrics, logging utilities
- `src/tasks`: task-specific library implementations
- `src/runner`: scripts to run single-task and comparison experiments
- `configs`: experiment and task configs
- `data/logs`: run-level outputs
- `results`: tables, figures, and summaries for report integration

## Quick Start
1. Create a virtual environment.
2. Install dependencies from `requirements.txt`.
3. Edit `configs/default.yaml` as needed.
4. Run `python src/runner/run_single_task.py --config configs/task_object.yaml`.
5. Run `python src/runner/run_comparison.py --config configs/default.yaml`.
