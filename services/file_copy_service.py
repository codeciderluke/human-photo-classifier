"""File copy service.

Creates the destination folders, preserves the source folder structure, and
deduplicates file names. Contains no detection logic.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


class FileCopyService:
    """Copies selected images into the destination folder."""

    def __init__(
        self,
        source_root: str | Path,
        destination_root: str | Path,
        preserve_structure: bool = True,
    ) -> None:
        """Create the copy service.

        Args:
            source_root: Top-level source folder, used as the base for structure preservation.
            destination_root: Top-level destination folder.
            preserve_structure: Whether to preserve the source subfolder structure.
        """
        self.source_root = Path(source_root)
        self.destination_root = Path(destination_root)
        self.preserve_structure = preserve_structure

    def copy(self, image_path: str | Path, subfolder: str = "") -> Path:
        """Copy the image into the destination folder and return the final path.

        Preserves the source folder structure and, if the target already exists,
        renames to ``name_1``, ``name_2`` etc. to avoid collisions.

        Args:
            image_path: Source image path to copy.
            subfolder: Category subfolder under the destination (e.g. ``with_face/male``).
                Empty string saves without a category folder.

        Returns:
            The actual destination file path that was written.
        """
        source = Path(image_path)
        target = self._resolve_target_path(source, subfolder)
        target.parent.mkdir(parents=True, exist_ok=True)
        target = self._deduplicate(target)

        shutil.copy2(source, target)
        logger.debug("Copied: %s -> %s", source, target)
        return target

    def _resolve_target_path(self, source: Path, subfolder: str = "") -> Path:
        """Compute the target path from the source path and category subfolder."""
        if self.preserve_structure:
            try:
                relative = source.relative_to(self.source_root)
            except ValueError:
                # File outside the source root: use the file name only.
                relative = Path(source.name)
        else:
            relative = Path(source.name)

        base = self.destination_root
        if subfolder:
            base = base / subfolder
        return base / relative

    @staticmethod
    def _deduplicate(target: Path) -> Path:
        """Return a non-colliding path if the target already exists."""
        if not target.exists():
            return target

        stem = target.stem
        suffix = target.suffix
        parent = target.parent
        index = 1
        while True:
            candidate = parent / f"{stem}_{index}{suffix}"
            if not candidate.exists():
                return candidate
            index += 1
