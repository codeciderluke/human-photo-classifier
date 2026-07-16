"""Shared test fixtures.

Injects a fake ``ultralytics`` module into ``sys.modules`` so tests run even
without the real package installed. Each test may freely mock the module's
``YOLO`` attribute.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

# Invalid (corrupt) image bytes.
CORRUPT_IMAGE_BYTES = b"this is definitely not a valid image file"


def write_valid_image(path: Path) -> Path:
    """Create a valid image file that passes the integrity check.

    With Pillow, writes a real valid image (integrity check active); otherwise
    writes arbitrary bytes (in which case the integrity check is skipped).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image

        Image.new("RGB", (8, 8), (20, 20, 20)).save(path)
    except ImportError:
        path.write_bytes(b"placeholder-image-bytes")
    return path


def write_corrupt_image(path: Path) -> Path:
    """Create a corrupt (undecodable) image file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(CORRUPT_IMAGE_BYTES)
    return path


@pytest.fixture
def fake_yolo(monkeypatch):
    """Inject a fake ``ultralytics.YOLO`` and return its mock factory.

    Returns:
        A ``MagicMock`` acting as ``YOLO``. Set the model instance it returns
        via ``fake_yolo.return_value``.
    """
    from unittest.mock import MagicMock

    yolo_factory = MagicMock(name="YOLO")

    module = types.ModuleType("ultralytics")
    module.YOLO = yolo_factory
    monkeypatch.setitem(sys.modules, "ultralytics", module)

    return yolo_factory


def make_box(class_id: int, confidence: float, xyxy: list[float]):
    """Create a mock imitating a YOLO ``box`` object."""
    from unittest.mock import MagicMock

    box = MagicMock()
    box.cls.item.return_value = class_id
    box.conf.item.return_value = confidence
    box.xyxy.__getitem__.return_value.tolist.return_value = xyxy
    return box


def make_result(boxes):
    """Create a mock imitating a YOLO result item (``result``)."""
    from unittest.mock import MagicMock

    result_item = MagicMock()
    result_item.boxes = boxes
    return result_item
