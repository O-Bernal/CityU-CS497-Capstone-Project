"""Helpers for resolving task runner selection from configuration."""


def select_library(task_cfg: dict) -> str:
    """Resolve a single library for single-task execution."""
    explicit = task_cfg.get("library")
    if explicit:
        return str(explicit)

    libraries = task_cfg.get("libraries", [])
    if libraries:
        return str(libraries[0])

    raise ValueError("Task config must define 'library' or non-empty 'libraries'.")
