"""Simple timing and throughput metrics collected during webcam runs."""

from dataclasses import dataclass, field
import time


@dataclass
class RunMetrics:
    """Collect per-frame timings and provide summary-level metrics."""

    frame_count: int = 0
    processing_times_ms: list[float] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)

    def record_frame(self, elapsed_ms: float) -> None:
        """Record elapsed processing time for one frame."""
        self.frame_count += 1
        self.processing_times_ms.append(elapsed_ms)

    def summary(self) -> dict:
        """Return aggregate stats such as FPS and average processing latency."""
        duration = max(time.time() - self.started_at, 1e-9)
        avg_ms = (
            sum(self.processing_times_ms) / len(self.processing_times_ms)
            if self.processing_times_ms
            else 0.0
        )
        return {
            "frame_count": self.frame_count,
            "duration_s": duration,
            "fps": self.frame_count / duration,
            "avg_processing_ms": avg_ms,
        }
