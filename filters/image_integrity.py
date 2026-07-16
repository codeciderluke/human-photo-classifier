"""Image integrity check.

Classifies image damage into three levels.

- ``OK``: Opens normally and decodes fully down to the pixels.
- ``PARTIAL``: The file structure (header) is valid, but the pixel data is
  truncated or partially damaged, so full decoding fails. (e.g. a JPEG cut off
  during transfer)
- ``CORRUPT``: Cannot be opened at all, or is not recognized as an image.

A pure utility with no dependency on the GUI/detection engine.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

OK = "ok"
PARTIAL = "partial"
CORRUPT = "corrupt"


def inspect_image(path: str | Path) -> str:
    """Determine an image's level of damage.

    Returns:
        One of ``"ok"``, ``"partial"``, ``"corrupt"``. If Pillow is not
        installed, the check is skipped and ``"ok"`` is returned.
    """
    try:
        import PIL.ImageFile
        from PIL import Image, UnidentifiedImageError
    except ImportError:
        return OK

    path = Path(path)

    # Step 1: structure/header validation. On failure, treat as fully corrupt (unusable).
    try:
        with Image.open(path) as image:
            image.verify()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        logger.debug("Fully corrupt image: %s", path)
        return CORRUPT

    # Step 2: disable truncated-image tolerance and fully decode down to the pixels.
    previous = PIL.ImageFile.LOAD_TRUNCATED_IMAGES
    PIL.ImageFile.LOAD_TRUNCATED_IMAGES = False
    try:
        with Image.open(path) as image:
            image.load()
    except (OSError, SyntaxError, ValueError):
        logger.debug("Partially damaged image: %s", path)
        return PARTIAL
    finally:
        PIL.ImageFile.LOAD_TRUNCATED_IMAGES = previous

    return OK
