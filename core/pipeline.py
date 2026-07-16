"""Image classification pipeline shared by the GUI and CLI.

Scans a folder, detects people (and optionally faces/gender), and copies images
into category subfolders. Has no GUI dependency: progress, logging, and stop
are plain callbacks.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from filters import (
    CORRUPT,
    PARTIAL,
    CorruptImageError,
    FaceAnalysisError,
    FaceModelLoadError,
    ImageAnalysisError,
    InsightFaceAnalyzer,
    ModelLoadError,
    YoloPersonFilter,
    inspect_image,
)
from services import (
    DAMAGED,
    OTHERS,
    FileCopyService,
    ImageScanner,
    build_subfolder,
)

logger = logging.getLogger(__name__)

ProgressCb = Callable[[int, int], None]
LogCb = Callable[[str], None]
StopCb = Callable[[], bool]


class PipelineError(RuntimeError):
    """Fatal error that aborts the whole job (bad folder or model load)."""


@dataclass
class ClassifyConfig:
    """Settings for a classification run."""

    source_folder: str
    destination_folder: str
    model_name: str = "yolo11n.pt"
    confidence: float = 0.20
    image_size: int = 960
    device: str = "cpu"
    recursive: bool = True
    detect_face: bool = False
    detect_gender: bool = False


@dataclass
class ClassifySummary:
    """Counts after a run completes."""

    total: int = 0
    with_person: int = 0
    others: int = 0
    damaged: int = 0
    corrupt: int = 0
    failed: int = 0
    copied: int = 0
    stopped: bool = False
    category_counts: dict[str, int] = field(default_factory=dict)


def run_classification(
    config: ClassifyConfig,
    *,
    on_progress: ProgressCb | None = None,
    on_log: LogCb | None = None,
    should_stop: StopCb | None = None,
) -> ClassifySummary:
    """Run the full scan -> detect -> classify -> copy pipeline.

    Args:
        config: Run settings.
        on_progress: Called as ``(current, total)`` after each image.
        on_log: Called with a human-readable message per image/step.
        should_stop: Polled before each image; return ``True`` to stop early.

    Returns:
        A summary of the run.

    Raises:
        PipelineError: A fatal error (bad folder, model load) aborted the run.
    """
    log: LogCb = on_log or (lambda _m: None)
    progress: ProgressCb = on_progress or (lambda _c, _t: None)
    stop: StopCb = should_stop or (lambda: False)

    # 1) Collect images
    try:
        images = ImageScanner(recursive=config.recursive).scan(config.source_folder)
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise PipelineError(f"Source folder error: {exc}") from exc

    summary = ClassifySummary(total=len(images))
    if summary.total == 0:
        log("No images to process.")
        return summary
    log(f"Found {summary.total} images.")

    # 2) Load the person filter once
    try:
        log(f"Loading model: {config.model_name} ({config.device})")
        person_filter = YoloPersonFilter(
            model_name=config.model_name,
            confidence=config.confidence,
            image_size=config.image_size,
            device=config.device,
        )
    except ValueError as exc:
        raise PipelineError(f"Configuration error: {exc}") from exc
    except ModelLoadError as exc:
        raise PipelineError(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise PipelineError(f"Unexpected error while loading model: {exc}") from exc
    log("Model loaded. Starting analysis.")

    # 2b) Load the face/gender analyzer only if needed
    classify_mode = config.detect_face or config.detect_gender
    face_analyzer = None
    if classify_mode:
        try:
            log("Loading face/gender model... (downloads on first run)")
            face_analyzer = InsightFaceAnalyzer(device=config.device)
        except FaceModelLoadError as exc:
            raise PipelineError(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise PipelineError(
                f"Unexpected error while loading face model: {exc}"
            ) from exc
        log("Face/gender model loaded.")

    copier = FileCopyService(config.source_folder, config.destination_folder)
    counts: Counter[str] = Counter()

    def do_copy(image_path: Path, subfolder: str) -> bool:
        try:
            copier.copy(image_path, subfolder=subfolder)
            counts[subfolder or "(unclassified)"] += 1
            return True
        except OSError as exc:
            log(f"[Copy failed] {image_path.name}: {exc}")
            return False

    # 3) Process each image
    for index, image_path in enumerate(images, start=1):
        if stop():
            summary.stopped = True
            break

        image_path = Path(image_path)
        name = image_path.name

        level = inspect_image(image_path)
        if level == CORRUPT or (level == PARTIAL and not classify_mode):
            summary.corrupt += 1
            log(f"[Corrupt - skipped] {name}")
            progress(index, summary.total)
            continue
        if level == PARTIAL:
            summary.damaged += 1
            if do_copy(image_path, DAMAGED):
                summary.copied += 1
                log(f"[Damaged -> others] {name}")
            progress(index, summary.total)
            continue

        try:
            result = person_filter.analyze(image_path)
        except CorruptImageError:
            summary.corrupt += 1
            log(f"[Corrupt - skipped] {name}")
            progress(index, summary.total)
            continue
        except ImageAnalysisError:
            summary.failed += 1
            log(f"[Analyze failed] {name}")
            progress(index, summary.total)
            continue
        except Exception as exc:  # noqa: BLE001 - skip per-image errors
            summary.failed += 1
            log(f"[Error] {name}: {exc}")
            progress(index, summary.total)
            continue

        if result.contains_person:
            summary.with_person += 1
            subfolder = _categorize(face_analyzer, config, image_path)
            prefix = f"Person {result.person_count}"
        elif classify_mode:
            summary.others += 1
            subfolder = OTHERS
            prefix = "Others"
        else:
            progress(index, summary.total)
            continue

        if do_copy(image_path, subfolder):
            summary.copied += 1
            label = subfolder.replace("/", " · ") if subfolder else "-"
            log(f"[{prefix} - {label}] {name}")

        progress(index, summary.total)

    summary.category_counts = dict(counts)
    return summary


def _categorize(face_analyzer, config: ClassifyConfig, image_path: Path) -> str:
    """Return the category subfolder from face/gender analysis (or empty)."""
    if face_analyzer is None:
        return ""
    try:
        face_result = face_analyzer.analyze(image_path)
        return build_subfolder(config.detect_face, config.detect_gender, face_result)
    except (FaceAnalysisError, CorruptImageError, OSError, ValueError):
        logger.exception("Face analysis failed, using unclassified: %s", image_path)
        return build_subfolder(
            config.detect_face, config.detect_gender, None, analysis_failed=True
        )
