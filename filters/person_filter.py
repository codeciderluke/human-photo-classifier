"""Person filter implementation based on Ultralytics YOLO.

Detects only the ``person`` class (class 0) of the COCO dataset. Not just full
bodies: an image counts as containing a person even when only part of the body
(upper/lower half) is detected.

This module does not depend on the GUI (PyQt5), and the model is loaded only
once at instance creation.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .base import ImagePersonFilter
from .models import PersonDetection, PersonFilterResult

logger = logging.getLogger(__name__)


class PersonFilterError(RuntimeError):
    """Top-level exception for errors during person detection."""


class ModelLoadError(PersonFilterError):
    """Detection model failed to load."""


class ImageAnalysisError(PersonFilterError):
    """Image analysis (inference) failed."""


class CorruptImageError(ImageAnalysisError):
    """Corrupt or unreadable image."""


class YoloPersonFilter(ImagePersonFilter):
    """Filter that decides whether a person is present using Ultralytics YOLO.

    The model is loaded exactly once in the constructor and is not reloaded on
    each ``analyze`` call.
    """

    #: ``person`` class ID in the COCO dataset.
    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        confidence: float = 0.20,
        image_size: int = 960,
        device: str | int = "cpu",
    ) -> None:
        """Create the filter and load the YOLO model.

        Args:
            model_name: File name or path of the YOLO model to load.
            confidence: Detection confidence threshold. ``0.0 < confidence <= 1.0``.
            image_size: Inference input image size in pixels. Must be positive.
            device: Inference device: ``"cpu"``, ``"cuda"``, ``"cuda:0"``, ``0``, etc.

        Raises:
            ValueError: A setting is out of the valid range.
            ModelLoadError: The YOLO model failed to load.
        """
        if not model_name or not model_name.strip():
            raise ValueError("model_name must not be empty.")

        if not 0.0 < confidence <= 1.0:
            raise ValueError("confidence must be greater than 0 and at most 1.")

        if image_size <= 0:
            raise ValueError("image_size must be positive.")

        self.model_name = model_name
        self.confidence = confidence
        self.image_size = image_size
        self.device = device

        # Import the heavy dependency (ultralytics) only when actually used,
        # to reduce overhead in tests and lightweight calls.
        from ultralytics import YOLO

        logger.info("Loading YOLO model: %s", model_name)
        try:
            self._model = YOLO(model_name)
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(
                f"Failed to load YOLO model: {model_name}"
            ) from exc
        logger.info("YOLO model loaded: %s", model_name)

    def analyze(self, image_path: str | Path) -> PersonFilterResult:
        """Analyze a single image and return the person detection result.

        Raises:
            FileNotFoundError: The image file does not exist.
            ValueError: The path is not a file.
            CorruptImageError: The image is corrupt and cannot be opened.
            ImageAnalysisError: An error occurred during inference.
        """
        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")

        if not path.is_file():
            raise ValueError(f"Not a file path: {path}")

        # Check image integrity before inference to reliably exclude corrupt files.
        self._verify_image_integrity(path)

        logger.debug("Analyzing image: %s", path)
        try:
            results = self._model.predict(
                source=str(path),
                conf=self.confidence,
                imgsz=self.image_size,
                classes=[self.PERSON_CLASS_ID],
                device=self.device,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Image analysis failed: %s", path)
            raise ImageAnalysisError(
                f"Image analysis failed: {path}"
            ) from exc

        detections = self._extract_person_detections(results)

        logger.debug("%d person(s) detected: %s", len(detections), path)
        return PersonFilterResult(
            image_path=path,
            contains_person=bool(detections),
            detections=tuple(detections),
        )

    @staticmethod
    def _verify_image_integrity(path: Path) -> None:
        """Check whether the image is corrupt to the point it cannot be opened.

        Only fully corrupt, unusable images (``CORRUPT``) raise an exception.
        Partially damaged images (``PARTIAL``) are allowed through so the upper
        layer can classify/handle them separately.

        Raises:
            CorruptImageError: The image is too corrupt to open.
        """
        from .image_integrity import CORRUPT, inspect_image

        if inspect_image(path) == CORRUPT:
            logger.warning("Excluding fully corrupt image: %s", path)
            raise CorruptImageError(
                f"Corrupt or unreadable image: {path}"
            )

    def _extract_person_detections(self, results) -> list[PersonDetection]:
        """Extract person detections from YOLO result objects.

        Even with the ``classes=[0]`` option, the class value is rechecked
        during conversion as a defensive measure.
        """
        detections: list[PersonDetection] = []

        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            for box in boxes:
                class_id = int(box.cls.item())
                if class_id != self.PERSON_CLASS_ID:
                    continue

                confidence = float(box.conf.item())
                coordinates = box.xyxy[0].tolist()

                detections.append(
                    PersonDetection(
                        confidence=confidence,
                        bounding_box=(
                            float(coordinates[0]),
                            float(coordinates[1]),
                            float(coordinates[2]),
                            float(coordinates[3]),
                        ),
                    )
                )

        return detections
