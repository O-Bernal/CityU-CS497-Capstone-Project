"""Minimal live webcam demo that draws face boxes in real time."""

import cv2

from src.core.camera import Camera


def open_camera_with_fallback(index: int = 0):
    """Try Windows DirectShow first, then default backend for webcam access."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if cap.isOpened():
        print(f"[INFO] Opened webcam index {index} with CAP_DSHOW")
        return cap

    cap.release()
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        print(f"[INFO] Opened webcam index {index} with default backend")
        return cap

    cap.release()
    return None


def main() -> None:
    """Open webcam, detect faces, and display annotated frames until quit."""
    camera = Camera(index=0)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)

    if detector.empty():
        raise RuntimeError(f"Could not load face cascade: {cascade_path}")

    print("[INFO] Starting webcam face demo...")
    print("[INFO] Press 'q' in the demo window to quit.")

    cap = open_camera_with_fallback(index=camera.index)
    if cap is None:
        raise RuntimeError(
            "Could not open webcam. Close other camera apps and verify camera privacy settings."
        )
    camera.cap = cap

    try:
        frame_failures = 0
        while True:
            ok, frame = camera.read()
            if not ok:
                frame_failures += 1
                if frame_failures >= 30:
                    raise RuntimeError(
                        "Webcam opened but no frames were received.\
                        Try another webcam index or close conflicting apps."
                    )
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (30, 200, 30), 2)

            cv2.putText(
                frame,
                "Face detection demo - press 'q' to quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.imshow("CS497 Webcam Demo", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Quit requested.")
                break
    finally:
        camera.close()
        cv2.destroyAllWindows()
        print("[INFO] Camera released and windows closed.")


if __name__ == "__main__":
    main()
