"""File-processing service package.

Handles image scanning and file copying, separate from the filtering modules.
"""

from .categorizer import DAMAGED, OTHERS, build_subfolder
from .file_copy_service import FileCopyService
from .image_scanner import ImageScanner

__all__ = [
    "ImageScanner",
    "FileCopyService",
    "build_subfolder",
    "OTHERS",
    "DAMAGED",
]
