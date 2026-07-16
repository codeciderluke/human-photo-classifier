"""Compute the destination subfolder path.

Given the face/gender detection options and the analysis result, decides the
subfolder (e.g. ``with_face/male``) where the image should be saved. Contains
no detection logic.
"""

from __future__ import annotations

from filters.models import GENDER_FEMALE, GENDER_MALE, FaceAnalysisResult

# Category folder names
FACE_PRESENT = "with_face"
FACE_ABSENT = "no_face"
GENDER_MALE_DIR = "male"
GENDER_FEMALE_DIR = "female"
UNCLASSIFIED = "unclassified"
OTHERS = "others"  # non-person photos
DAMAGED = "others/damaged"  # damaged photos (under others)


def build_subfolder(
    detect_face: bool,
    detect_gender: bool,
    face_result: FaceAnalysisResult | None,
    analysis_failed: bool = False,
) -> str:
    """Compute the relative subfolder path where the image should be saved.

    Rules:
        - Both options off -> empty string (no category, keep source structure).
        - Analysis failed -> ``unclassified``.
        - Face detection only -> ``with_face`` / ``no_face``.
        - Gender detection only -> ``male`` / ``female`` / ``unclassified``.
        - Both -> ``with_face/male`` etc. If no face, ``no_face``.

    Args:
        detect_face: Whether face detection is enabled.
        detect_gender: Whether gender detection is enabled.
        face_result: Face analysis result, or ``None`` if it failed or was skipped.
        analysis_failed: Whether face analysis failed with an error.

    Returns:
        The ``/``-separated relative subfolder path, or empty string if uncategorized.
    """
    if not detect_face and not detect_gender:
        return ""

    if analysis_failed or face_result is None:
        return UNCLASSIFIED

    has_face = face_result.has_face
    parts: list[str] = []

    if detect_face:
        parts.append(FACE_PRESENT if has_face else FACE_ABSENT)

    if detect_gender:
        if not has_face:
            # Gender cannot be determined without a face.
            if not detect_face:
                parts.append(UNCLASSIFIED)
            # If face detection is also on, it is already sorted as 'no_face'.
        else:
            parts.append(_gender_dir(face_result.dominant_gender))

    return "/".join(parts)


def _gender_dir(gender: str | None) -> str:
    if gender == GENDER_MALE:
        return GENDER_MALE_DIR
    if gender == GENDER_FEMALE:
        return GENDER_FEMALE_DIR
    return UNCLASSIFIED
