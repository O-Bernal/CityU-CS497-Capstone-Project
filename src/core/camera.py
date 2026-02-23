"""Shared webcam wrapper for opening, reading, and releasing camera streams."""


class Camera:
    """Minimal camera abstraction used by runners and demos."""

    def __init__(self, index: int = 0):
        """Store the camera index and initialize capture handle state."""
        self.index = index
        self.cap = None

    def open(self) -> None:
        """Open the configured webcam index with Windows-friendly backend fallback."""
        import cv2

        # Windows MSMF can be very slow/unstable on some webcams; prefer DirectShow first.
        cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if cap.isOpened():
            self.cap = cap
            return

        cap.release()
        cap = cv2.VideoCapture(self.index)
        if cap.isOpened():
            self.cap = cap
            return

        cap.release()
        raise RuntimeError(f"Unable to open webcam index {self.index}")

    def read(self):
        """Read a single frame from the active webcam stream."""
        if self.cap is None:
            raise RuntimeError("Camera not opened")
        ok, frame = self.cap.read()
        return ok, frame

    def close(self) -> None:
        """Release the webcam stream if it is open."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
