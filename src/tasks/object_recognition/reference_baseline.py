"""Shared helpers for the reduced-scope object-recognition reference baseline."""

from __future__ import annotations

from pathlib import Path


KNOWN_OBJECT_LABELS = ("pen", "key", "cup", "glasses")
IMAGE_PATTERNS = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REFERENCE_ROOT = REPO_ROOT / "data" / "object_dataset"
DEFAULT_EMBEDDER_MODEL = REPO_ROOT / "models" / "mediapipe" / "efficientdet_lite0.tflite"


def infer_reference_label(path: Path, reference_root: Path) -> str | None:
    """Infer an object label from a reference image path."""
    relative = path.relative_to(reference_root)
    for part in relative.parts[:-1]:
        lowered = part.lower()
        if lowered in KNOWN_OBJECT_LABELS:
            return lowered

    lowered_name = path.stem.lower()
    for label in KNOWN_OBJECT_LABELS:
        if lowered_name == label or lowered_name.startswith(f"{label}_") or lowered_name.startswith(f"{label}-"):
            return label
    return None


def collect_reference_paths(reference_root: Path | None = None) -> tuple[dict[str, list[Path]], str | None]:
    """Collect reference image paths for the fixed small-object baseline."""
    root = reference_root or DEFAULT_REFERENCE_ROOT
    if not root.exists():
        return {}, (
            f"Object reference dataset is missing: {root}. "
            "Add labeled images for pen, key, cup, or glasses under data/object_dataset."
        )

    references: dict[str, list[Path]] = {label: [] for label in KNOWN_OBJECT_LABELS}
    for pattern in IMAGE_PATTERNS:
        for path in sorted(root.rglob(pattern)):
            if not path.is_file():
                continue
            label = infer_reference_label(path, root)
            if label is not None:
                references[label].append(path)

    populated = {label: paths for label, paths in references.items() if paths}
    if populated:
        return populated, None

    return {}, (
        f"No usable object reference images were found in {root}. "
        "Expected labeled images for pen, key, cup, or glasses in subfolders or filenames."
    )


def build_candidate_views(frame, *, center_fraction: float = 0.7) -> list[dict]:
    """Build full-frame and center-crop candidate regions for handheld object matching."""
    height, width = frame.shape[:2]
    views = [{"image": frame, "bbox": [0, 0, int(width), int(height)]}]

    crop_width = max(1, int(width * center_fraction))
    crop_height = max(1, int(height * center_fraction))
    x = max(0, (width - crop_width) // 2)
    y = max(0, (height - crop_height) // 2)
    cropped = frame[y : y + crop_height, x : x + crop_width]
    views.insert(0, {"image": cropped, "bbox": [int(x), int(y), int(crop_width), int(crop_height)]})
    return views
