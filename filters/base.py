"""Abstract interface for person filters.

Implementing this abstract class lets the filter be swapped for detection
engines other than YOLO (OpenCV DNN, ONNX Runtime, external APIs, etc.).
Upper layers depend only on this interface, not the concrete implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

from .models import PersonFilterResult


class ImagePersonFilter(ABC):
    """Abstract interface for a filter that decides whether an image contains a person."""

    @abstractmethod
    def analyze(self, image_path: str | Path) -> PersonFilterResult:
        """Analyze an image and return the person detection result."""
        raise NotImplementedError

    def contains_person(self, image_path: str | Path) -> bool:
        """Return whether the image contains at least one person."""
        return self.analyze(image_path).contains_person

    def analyze_many(
        self,
        image_paths: Iterable[str | Path],
        should_stop: Callable[[], bool] | None = None,
    ) -> Iterator[PersonFilterResult]:
        """Analyze multiple images sequentially.

        Progress and stop decisions are the caller's responsibility. Instead of
        taking a GUI object like a PyQt signal, stopping is passed as a plain callback.

        Args:
            should_stop: Callback that, when it returns ``True``, aborts iteration.

        Yields:
            The filter result for each image.
        """
        for image_path in image_paths:
            if should_stop is not None and should_stop():
                return
            yield self.analyze(image_path)
