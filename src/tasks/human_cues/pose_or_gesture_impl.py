"""Placeholder human-cue pose/gesture task interface."""

from src.tasks.interface import TaskResult, make_result


def run(frame) -> TaskResult:
    """Process one frame and return coarse pose and gesture metadata."""
    # Placeholder implementation for human-cues pipeline.
    return make_result(
        task="human_cues",
        library="mediapipe",
        outputs={"pose": None, "gesture": None},
    )
