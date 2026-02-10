"""Configuration loading helpers for YAML-based experiment settings."""

from pathlib import Path
import yaml


def load_config(path: str) -> dict:
    """Load and return a YAML config file as a dictionary."""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
