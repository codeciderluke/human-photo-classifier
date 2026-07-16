"""Unit tests for classification folder path building (``build_subfolder``)."""

from __future__ import annotations

from pathlib import Path

from filters.models import (
    GENDER_FEMALE,
    GENDER_MALE,
    FaceAnalysisResult,
    FaceInfo,
)
from services.categorizer import build_subfolder


def _result(has_face: bool, gender: str | None = None) -> FaceAnalysisResult:
    faces: tuple[FaceInfo, ...] = ()
    if has_face:
        faces = (
            FaceInfo(
                confidence=0.99,
                bounding_box=(0.0, 0.0, 100.0, 100.0),
                gender=gender,
                age=30,
            ),
        )
    return FaceAnalysisResult(Path("x.jpg"), has_face, faces)


# ------------------------------------------------------------ options off

def test_no_options_returns_empty():
    assert build_subfolder(False, False, _result(True, GENDER_MALE)) == ""


# ------------------------------------------------------------- face only

def test_face_only_present():
    assert build_subfolder(True, False, _result(True)) == "with_face"


def test_face_only_absent():
    assert build_subfolder(True, False, _result(False)) == "no_face"


# ----------------------------------------------------------- gender only

def test_gender_only_male():
    assert build_subfolder(False, True, _result(True, GENDER_MALE)) == "male"


def test_gender_only_female():
    assert build_subfolder(False, True, _result(True, GENDER_FEMALE)) == "female"


def test_gender_only_no_face_is_unclassified():
    assert build_subfolder(False, True, _result(False)) == "unclassified"


def test_gender_only_unknown_gender_is_unclassified():
    assert build_subfolder(False, True, _result(True, None)) == "unclassified"


# ---------------------------------------------------------------- both

def test_both_face_and_male():
    assert build_subfolder(True, True, _result(True, GENDER_MALE)) == "with_face/male"


def test_both_face_and_female():
    got = build_subfolder(True, True, _result(True, GENDER_FEMALE))
    assert got == "with_face/female"


def test_both_no_face_is_face_absent_only():
    # No face: classify only as 'no_face' without a gender subfolder
    assert build_subfolder(True, True, _result(False)) == "no_face"


def test_both_face_unknown_gender():
    got = build_subfolder(True, True, _result(True, None))
    assert got == "with_face/unclassified"


# ------------------------------------------------------------ analysis failed

def test_analysis_failed_is_unclassified():
    assert build_subfolder(True, True, None, analysis_failed=True) == "unclassified"


def test_none_result_is_unclassified():
    assert build_subfolder(True, False, None) == "unclassified"
