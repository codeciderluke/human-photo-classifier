"""Image file scanning service.

Handles extension filtering and subfolder search. Does not depend on any
detection engine or GUI.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

logger = logging.getLogger(__name__)

#: Default image extensions to scan.
DEFAULT_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
)


class ImageScanner:
    """Collects image files from a folder."""

    def __init__(
        self,
        extensions: frozenset[str] | set[str] | None = None,
        recursive: bool = True,
    ) -> None:
        """Create the scanner.

        Args:
            extensions: Set of extensions to collect. Uses the default if ``None``.
            recursive: Whether to search subfolders.
        """
        source = extensions if extensions is not None else DEFAULT_IMAGE_EXTENSIONS
        # Normalize extensions to lowercase with a leading dot.
        self.extensions = frozenset(self._normalize_ext(ext) for ext in source)
        self.recursive = recursive

    @staticmethod
    def _normalize_ext(ext: str) -> str:
        ext = ext.strip().lower()
        if not ext.startswith("."):
            ext = "." + ext
        return ext

    def scan(self, folder: str | Path) -> list[Path]:
        """Return a sorted list of image files in the folder.

        Args:
            folder: Folder path to scan.

        Returns:
            Sorted list of matching image file paths.

        Raises:
            FileNotFoundError: If the folder does not exist.
            NotADirectoryError: If the path is not a folder.
        """
        return sorted(self.iter_scan(folder))

    def iter_scan(self, folder: str | Path) -> Iterator[Path]:
        """Lazily iterate over image files in the folder.

        Args:
            folder: Folder path to scan.

        Yields:
            Matching image file paths.

        Raises:
            FileNotFoundError: If the folder does not exist.
            NotADirectoryError: If the path is not a folder.
        """
        root = Path(folder)

        if not root.exists():
            raise FileNotFoundError(f"Folder not found: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"Not a folder path: {root}")

        pattern = "**/*" if self.recursive else "*"
        for path in root.glob(pattern):
            if path.is_file() and path.suffix.lower() in self.extensions:
                yield path
