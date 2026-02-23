"""Task registry for resolving task+library pairs into runnable callables."""

from importlib import import_module
from typing import Callable

from src.tasks.interface import TaskResult

TASK_MODULES = {
    ("object_recognition", "opencv"): "src.tasks.object_recognition.opencv_impl",
    ("object_recognition", "mediapipe"): "src.tasks.object_recognition.mediapipe_impl",
    ("ocr", "tesseract"): "src.tasks.ocr.tesseract_impl",
    ("ocr", "easyocr"): "src.tasks.ocr.easyocr_impl",
    ("human_cues", "opencv"): "src.tasks.human_cues.face_detection_impl",
    ("human_cues", "mediapipe"): "src.tasks.human_cues.pose_or_gesture_impl",
}


def get_task_runner(task_name: str, library_name: str) -> Callable[[object], TaskResult]:
    """Load and return the run(frame) function for a task-library combination."""
    key = (task_name, library_name)
    module_path = TASK_MODULES.get(key)
    if module_path is None:
        supported = ", ".join(f"{task}/{lib}" for task, lib in sorted(TASK_MODULES))
        raise ValueError(
            f"Unsupported task/library: {task_name}/{library_name}. Supported: {supported}"
        )

    module = import_module(module_path)
    runner = getattr(module, "run", None)
    if not callable(runner):
        raise RuntimeError(f"Module {module_path} does not expose callable run(frame)")
    return runner
