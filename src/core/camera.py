"""Shared webcam wrapper for opening, configuring, and reading camera streams."""


class Camera:
    """Minimal camera abstraction used by runners and demos."""

    def __init__(self, index: int = 0, width: int | None = None, height: int | None = None):
        """Store the camera index and optional target resolution."""
        self.index = index
        self.width = int(width) if width is not None else None
        self.height = int(height) if height is not None else None
        self.cap = None

    def _apply_resolution(self, cap) -> None:
        """Apply configured width and height to an opened capture device."""
        import cv2

        if self.width is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def open(self) -> None:
        """Open the configured webcam index with Windows-friendly backend fallback."""
        import cv2

        # Windows MSMF can be very slow/unstable on some webcams; prefer DirectShow first.
        cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if cap.isOpened():
            self._apply_resolution(cap)
            self.cap = cap
            return

        cap.release()
        cap = cv2.VideoCapture(self.index)
        if cap.isOpened():
            self._apply_resolution(cap)
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

    def actual_resolution(self) -> tuple[int | None, int | None]:
        """Return the resolution reported by the active capture device."""
        import cv2

        if self.cap is None:
            return self.width, self.height

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or self.width
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or self.height
        return width, height

    def close(self) -> None:
        """Release the webcam stream if it is open."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
