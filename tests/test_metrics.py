"""Tests for run-metrics aggregation behavior."""

from src.core.metrics import RunMetrics


def test_metrics_summary_has_expected_fields():
    """Ensure summary output exposes core fields used by reporting."""
    m = RunMetrics()
    m.record_frame(10.0)
    out = m.summary()
    assert "frame_count" in out
    assert "fps" in out
    assert "avg_processing_ms" in out
