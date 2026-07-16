"""Unit tests for ``YoloPersonFilter``.

Uses a fake ``ultralytics.YOLO`` to verify detection logic, result conversion,
and exception handling without real model downloads or inference.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from filters import (
    CorruptImageError,
    ImageAnalysisError,
    ModelLoadError,
    PersonFilterResult,
    YoloPersonFilter,
)
from tests.conftest import (
    make_box,
    make_result,
    write_corrupt_image,
    write_valid_image,
)


@pytest.fixture
def image_file(tmp_path: Path) -> Path:
    return write_valid_image(tmp_path / "sample.png")


# ------------------------------------------------------------- model loading

def test_model_is_created_once_in_constructor(fake_yolo):
    YoloPersonFilter(model_name="yolo11n.pt")
    fake_yolo.assert_called_once_with("yolo11n.pt")


def test_analyze_does_not_reload_model(fake_yolo, image_file):
    model = fake_yolo.return_value
    model.predict.return_value = [make_result([])]

    person_filter = YoloPersonFilter()
    person_filter.analyze(image_file)
    person_filter.analyze(image_file)

    # The model must be created only once.
    fake_yolo.assert_called_once()
    assert model.predict.call_count == 2


def test_model_load_failure_raises_model_load_error(fake_yolo):
    fake_yolo.side_effect = RuntimeError("weights not found")

    with pytest.raises(ModelLoadError):
        YoloPersonFilter()


# ----------------------------------------------------------- predict options

def test_predict_called_with_expected_options(fake_yolo, image_file):
    model = fake_yolo.return_value
    model.predict.return_value = [make_result([])]

    person_filter = YoloPersonFilter(
        confidence=0.35, image_size=1280, device="cuda:0"
    )
    person_filter.analyze(image_file)

    _, kwargs = model.predict.call_args
    assert kwargs["conf"] == 0.35
    assert kwargs["imgsz"] == 1280
    assert kwargs["classes"] == [0]
    assert kwargs["device"] == "cuda:0"
    assert kwargs["verbose"] is False
    assert kwargs["source"] == str(image_file)


# --------------------------------------------------------- result evaluation

def test_no_detection_returns_contains_person_false(fake_yolo, image_file):
    model = fake_yolo.return_value
    model.predict.return_value = [make_result([])]

    person_filter = YoloPersonFilter()
    result = person_filter.analyze(image_file)

    assert isinstance(result, PersonFilterResult)
    assert result.contains_person is False
    assert result.detections == ()


def test_person_detection_returns_contains_person_true(fake_yolo, image_file):
    box = make_box(class_id=0, confidence=0.91, xyxy=[10.0, 20.0, 100.0, 200.0])
    model = fake_yolo.return_value
    model.predict.return_value = [make_result([box])]

    person_filter = YoloPersonFilter()
    result = person_filter.analyze(image_file)

    assert result.contains_person is True
    assert len(result.detections) == 1
    assert result.detections[0].confidence == pytest.approx(0.91)
    assert result.detections[0].bounding_box == (10.0, 20.0, 100.0, 200.0)


def test_multiple_persons_all_returned(fake_yolo, image_file):
    boxes = [
        make_box(0, 0.90, [0, 0, 10, 10]),
        make_box(0, 0.80, [20, 20, 30, 30]),
        make_box(0, 0.70, [40, 40, 50, 50]),
    ]
    model = fake_yolo.return_value
    model.predict.return_value = [make_result(boxes)]

    person_filter = YoloPersonFilter()
    result = person_filter.analyze(image_file)

    assert result.person_count == 3
    assert result.contains_person is True


def test_non_person_class_is_ignored_defensively(fake_yolo, image_file):
    # Even with classes=[0], defensively re-check the class value.
    boxes = [make_box(class_id=15, confidence=0.99, xyxy=[0, 0, 5, 5])]
    model = fake_yolo.return_value
    model.predict.return_value = [make_result(boxes)]

    person_filter = YoloPersonFilter()
    result = person_filter.analyze(image_file)

    assert result.contains_person is False


def test_contains_person_convenience_method(fake_yolo, image_file):
    box = make_box(0, 0.75, [1, 2, 3, 4])
    model = fake_yolo.return_value
    model.predict.return_value = [make_result([box])]

    person_filter = YoloPersonFilter()
    assert person_filter.contains_person(image_file) is True


# ------------------------------------------------------------- exceptions

def test_missing_file_raises_file_not_found(fake_yolo, tmp_path):
    fake_yolo.return_value.predict.return_value = [make_result([])]
    person_filter = YoloPersonFilter()

    with pytest.raises(FileNotFoundError):
        person_filter.analyze(tmp_path / "does_not_exist.jpg")


def test_directory_path_raises_value_error(fake_yolo, tmp_path):
    person_filter = YoloPersonFilter()

    with pytest.raises(ValueError):
        person_filter.analyze(tmp_path)


def test_inference_failure_raises_image_analysis_error(fake_yolo, image_file):
    model = fake_yolo.return_value
    model.predict.side_effect = RuntimeError("cuda oom")

    person_filter = YoloPersonFilter()
    with pytest.raises(ImageAnalysisError):
        person_filter.analyze(image_file)


# ------------------------------------------------------------ corrupt image

def test_corrupt_image_raises_corrupt_image_error(fake_yolo, tmp_path):
    pytest.importorskip("PIL")  # integrity check requires Pillow

    corrupt = write_corrupt_image(tmp_path / "broken.jpg")
    model = fake_yolo.return_value

    person_filter = YoloPersonFilter()
    with pytest.raises(CorruptImageError):
        person_filter.analyze(corrupt)

    # A corrupt image must be excluded and never reach inference.
    model.predict.assert_not_called()


def test_corrupt_image_error_is_analysis_error_subclass():
    # Keep the hierarchy so workers can also catch it as ImageAnalysisError.
    assert issubclass(CorruptImageError, ImageAnalysisError)


# ------------------------------------------------------- constructor checks

@pytest.mark.parametrize("bad_conf", [0.0, -0.1, 1.5])
def test_invalid_confidence_rejected(fake_yolo, bad_conf):
    with pytest.raises(ValueError):
        YoloPersonFilter(confidence=bad_conf)


def test_empty_model_name_rejected(fake_yolo):
    with pytest.raises(ValueError):
        YoloPersonFilter(model_name="   ")


@pytest.mark.parametrize("bad_size", [0, -100])
def test_invalid_image_size_rejected(fake_yolo, bad_size):
    with pytest.raises(ValueError):
        YoloPersonFilter(image_size=bad_size)


# -------------------------------------------------------------- batch analysis

def test_analyze_many_yields_all(fake_yolo, tmp_path):
    model = fake_yolo.return_value
    model.predict.return_value = [make_result([])]

    files = [write_valid_image(tmp_path / f"img_{i}.png") for i in range(3)]

    person_filter = YoloPersonFilter()
    results = list(person_filter.analyze_many(files))
    assert len(results) == 3


def test_analyze_many_respects_stop_callback(fake_yolo, tmp_path):
    model = fake_yolo.return_value
    model.predict.return_value = [make_result([])]

    files = [write_valid_image(tmp_path / f"img_{i}.png") for i in range(5)]

    person_filter = YoloPersonFilter()

    calls = {"n": 0}

    def should_stop() -> bool:
        calls["n"] += 1
        return calls["n"] > 2  # stop after processing two

    results = list(person_filter.analyze_many(files, should_stop=should_stop))
    assert len(results) == 2
