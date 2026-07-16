"""Unit tests for ``InsightFaceAnalyzer``'s engine-independent logic.

Excludes parts needing model loading (``analyze``, ``__init__``); verifies only
pure logic such as device-to-ctx conversion and gender mapping.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from filters.face_analyzer import InsightFaceAnalyzer
from filters.models import GENDER_FEMALE, GENDER_MALE


# --------------------------------------------------- device -> ctx_id mapping

@pytest.mark.parametrize(
    "device, expected",
    [
        ("cpu", -1),
        ("cuda", 0),
        ("cuda:0", 0),
        ("cuda:1", 1),
        ("0", 0),
        (0, 0),
        (1, 1),
    ],
)
def test_device_to_ctx(device, expected):
    assert InsightFaceAnalyzer._device_to_ctx(device) == expected


# ------------------------------------------------------------ gender mapping

def test_gender_from_sex_attribute():
    assert InsightFaceAnalyzer._gender_of(SimpleNamespace(sex="M")) == GENDER_MALE
    assert InsightFaceAnalyzer._gender_of(SimpleNamespace(sex="F")) == GENDER_FEMALE


def test_gender_from_gender_int():
    # No sex attribute, only gender(int): 1=male, 0=female
    assert InsightFaceAnalyzer._gender_of(SimpleNamespace(gender=1)) == GENDER_MALE
    assert InsightFaceAnalyzer._gender_of(SimpleNamespace(gender=0)) == GENDER_FEMALE


def test_gender_unknown_returns_none():
    assert InsightFaceAnalyzer._gender_of(SimpleNamespace()) is None
