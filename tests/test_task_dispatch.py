"""Tests for task result schema and task-library dispatch."""

from src.runner.task_selection import select_library
from src.tasks.interface import make_result
from src.tasks.registry import get_task_runner


def test_make_result_has_standard_keys():
    """Shared task-result helper should return the expected schema."""
    out = make_result(task="ocr", library="tesseract", outputs={"text": "hi"})
    assert set(out.keys()) == {"task", "library", "ok", "outputs", "error"}
    assert out["task"] == "ocr"
    assert out["library"] == "tesseract"
    assert out["ok"] is True


def test_select_library_prefers_explicit_library():
    """Single-task runner should use task.library when provided."""
    cfg = {"library": "mediapipe", "libraries": ["opencv"]}
    assert select_library(cfg) == "mediapipe"


def test_select_library_falls_back_to_first_list_item():
    """Single-task runner should use first task.libraries entry if needed."""
    cfg = {"libraries": ["easyocr", "tesseract"]}
    assert select_library(cfg) == "easyocr"


def test_registry_returns_callable_runner():
    """Task registry should return a callable run function for supported combos."""
    runner = get_task_runner("object_recognition", "opencv")
    assert callable(runner)
