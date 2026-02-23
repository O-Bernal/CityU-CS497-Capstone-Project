"""Shared task result contract for task-library adapters."""

from typing import Any, TypedDict


class Detection(TypedDict, total=False):
    """Standardized single-detection record."""

    label: str
    confidence: float | None
    bbox: list[int]


class TaskResult(TypedDict):
    """Standardized per-frame output returned by every task implementation."""

    task: str
    library: str
    ok: bool
    outputs: dict[str, Any]
    error: str | None


def make_result(
    *,
    task: str,
    library: str,
    outputs: dict[str, Any] | None = None,
    ok: bool = True,
    error: str | None = None,
) -> TaskResult:
    """Create a TaskResult object with a consistent schema."""
    return {
        "task": task,
        "library": library,
        "ok": ok,
        "outputs": outputs or {},
        "error": error,
    }
