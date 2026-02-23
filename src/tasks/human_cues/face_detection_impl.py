"""Placeholder human-cue face-detection task interface."""

from src.tasks.interface import TaskResult, make_result


def run(frame) -> TaskResult:
    """Process one frame and return detected face metadata."""
    # Placeholder implementation for human-cues pipeline.
    return make_result(
        task="human_cues",
        library="opencv",
        outputs={"faces": []},
    )
