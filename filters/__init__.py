"""Person photo filtering package.

Provides pure person detection/selection modules decoupled from the GUI layer.
Has no dependency on GUI frameworks like PyQt5, so it can be reused in CLI,
batch, server API, and test environments.
"""

from .base import ImagePersonFilter
from .face_analyzer import (
    FaceAnalysisError,
    FaceModelLoadError,
    ImageFaceAnalyzer,
    InsightFaceAnalyzer,
)
from .image_integrity import CORRUPT, OK, PARTIAL, inspect_image
from .models import (
    FaceAnalysisResult,
    FaceInfo,
    PersonDetection,
    PersonFilterResult,
)
from .person_filter import (
    CorruptImageError,
    ImageAnalysisError,
    ModelLoadError,
    PersonFilterError,
    YoloPersonFilter,
)

__all__ = [
    "ImagePersonFilter",
    "PersonDetection",
    "PersonFilterResult",
    "YoloPersonFilter",
    "PersonFilterError",
    "ModelLoadError",
    "ImageAnalysisError",
    "CorruptImageError",
    "ImageFaceAnalyzer",
    "InsightFaceAnalyzer",
    "FaceAnalysisResult",
    "FaceInfo",
    "FaceAnalysisError",
    "FaceModelLoadError",
    "inspect_image",
    "OK",
    "PARTIAL",
    "CORRUPT",
]
