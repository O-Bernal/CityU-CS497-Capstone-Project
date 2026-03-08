"""Tests for YAML configuration loading."""

from src.core.config import load_config


def test_load_config_reads_yaml():
    """Confirm base config can be loaded into a dictionary."""
    cfg = load_config("configs/default.yaml")
    assert "camera" in cfg
    assert "comparison" in cfg


def test_load_focused_comparison_configs():
    """Face and OCR configs should both load into dictionaries."""
    face_cfg = load_config("configs/face_comparison.yaml")
    ocr_cfg = load_config("configs/task_ocr.yaml")

    assert face_cfg["comparison"]["repeats"] == 3
    assert face_cfg["comparison"]["conditions"] == [
        "bright_clean",
        "normal_cluttered",
        "dim_cluttered_far",
    ]
    assert ocr_cfg["dataset"]["image_dir"] == "data/ocr_dataset"
