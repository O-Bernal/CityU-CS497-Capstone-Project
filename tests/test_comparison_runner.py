"""Tests for comparison matrix expansion behavior."""

from src.runner.run_comparison import _expand_matrix


def test_expand_matrix_uses_task_specific_libraries():
    """Comparison matrix should honor task_libraries mapping and repeats."""
    cfg = {
        "task_libraries": {"object_recognition": ["opencv", "mediapipe"]},
        "conditions": ["bright"],
        "repeats": 2,
    }

    out = _expand_matrix(cfg)
    assert len(out) == 4
    assert ("object_recognition", "opencv", "bright", 1) in out
    assert ("object_recognition", "mediapipe", "bright", 2) in out


def test_expand_matrix_skips_unsupported_pairs():
    """Unsupported task/library combinations should be excluded from the plan."""
    cfg = {
        "tasks": ["ocr"],
        "libraries": ["opencv"],
        "conditions": ["base"],
        "repeats": 1,
    }

    out = _expand_matrix(cfg)
    assert out == []
