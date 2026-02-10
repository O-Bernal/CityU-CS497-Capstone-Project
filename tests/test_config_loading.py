"""Tests for YAML configuration loading."""

from src.core.config import load_config


def test_load_config_reads_yaml():
    """Confirm base config can be loaded into a dictionary."""
    cfg = load_config("configs/default.yaml")
    assert "camera" in cfg
    assert "comparison" in cfg
