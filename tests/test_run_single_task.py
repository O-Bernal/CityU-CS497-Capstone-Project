"""Tests for flat run-record generation used by face comparison logging."""

from collections import Counter

from src.runner.run_single_task import _build_run_record


def test_build_run_record_has_paper_fields():
    """Run records should expose the face-comparison fields needed for CSV export."""
    record = _build_run_record(
        task_name="human_cues",
        library_name="opencv",
        condition="bright_clean",
        repeat=2,
        summary={"frame_count": 30, "duration_s": 5.0, "fps": 6.0, "avg_processing_ms": 18.4},
        open_ms=120.0,
        warmup_frames=15,
        frames_with_detection=24,
        label_counts=Counter({"face": 20, "person": 4}),
        failed_frames=1,
        avg_confidence=None,
        resolution=(640, 480),
        video_path=None,
    )

    assert record["library"] == "opencv"
    assert record["condition"] == "bright_clean"
    assert record["repeat"] == 2
    assert record["frames_processed"] == 30
    assert record["duration_s"] == 5.0
    assert record["fps"] == 6.0
    assert record["avg_processing_ms"] == 18.4
    assert record["webcam_open_ms"] == 120.0
    assert record["frames_with_detection"] == 24
    assert record["detection_rate"] == 0.8
    assert record["label_counts"] == {"face": 20, "person": 4}
    assert record["verdict"] is None
    assert record["notes"] is None
