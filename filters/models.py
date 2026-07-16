"""Result data models for the filtering module.

Defines only pure value objects that do not depend on the GUI or detection
engine. Frozen dataclasses are used to prevent side effects while passing results.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

# Gender value constants (engine-independent internal representation)
GENDER_MALE = "male"
GENDER_FEMALE = "female"


@dataclass(frozen=True)
class PersonDetection:
    """A single person detection.

    Attributes:
        confidence: Detection confidence (0.0 to 1.0).
        bounding_box: Box coordinates as ``(x1, y1, x2, y2)``.
    """

    confidence: float
    bounding_box: Tuple[float, float, float, float]


@dataclass(frozen=True)
class PersonFilterResult:
    """Person filtering result for a single image.

    Attributes:
        image_path: Path of the analyzed image.
        contains_person: Whether at least one person is present.
        detections: List of detected people.
    """

    image_path: Path
    contains_person: bool
    detections: Tuple[PersonDetection, ...]

    @property
    def person_count(self) -> int:
        """Number of detected people."""
        return len(self.detections)


@dataclass(frozen=True)
class FaceInfo:
    """A single face detection.

    Attributes:
        confidence: Face detection confidence (0.0 to 1.0).
        bounding_box: Box coordinates as ``(x1, y1, x2, y2)``.
        gender: Estimated gender: ``"male"``, ``"female"``, or ``None`` if unknown.
        age: Estimated age, or ``None`` if unknown.
    """

    confidence: float
    bounding_box: Tuple[float, float, float, float]
    gender: Optional[str]
    age: Optional[int]


@dataclass(frozen=True)
class FaceAnalysisResult:
    """Face/gender analysis result for a single image.

    Attributes:
        image_path: Path of the analyzed image.
        has_face: Whether at least one face was detected.
        faces: List of detected faces.
    """

    image_path: Path
    has_face: bool
    faces: Tuple[FaceInfo, ...]

    @property
    def face_count(self) -> int:
        """Number of detected faces."""
        return len(self.faces)

    @property
    def dominant_face(self) -> Optional[FaceInfo]:
        """Return the largest face (the apparent main subject)."""
        if not self.faces:
            return None

        def area(face: FaceInfo) -> float:
            x1, y1, x2, y2 = face.bounding_box
            return abs(x2 - x1) * abs(y2 - y1)

        return max(self.faces, key=area)

    @property
    def dominant_gender(self) -> Optional[str]:
        """Estimated gender of the dominant face; ``None`` if no face or unknown."""
        face = self.dominant_face
        return face.gender if face is not None else None
