"""Unit tests for the shared classification pipeline (extract mode).

Face/gender (classify) mode needs the real InsightFace model, so it is covered
by end-to-end checks rather than unit tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core import ClassifyConfig, PipelineError, run_classification
from tests.conftest import (
    make_box,
    make_result,
    write_corrupt_image,
    write_valid_image,
)

pytest.importorskip("PIL")


def _config(src: Path, dst: Path, **kw) -> ClassifyConfig:
    return ClassifyConfig(source_folder=str(src), destination_folder=str(dst), **kw)


def test_extracts_only_person_photos(fake_yolo, tmp_path: Path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    write_valid_image(src / "a.png")  # person
    write_valid_image(src / "b.png")  # no person

    model = fake_yolo.return_value
    box = make_box(0, 0.9, [0, 0, 10, 10])
    model.predict.side_effect = [[make_result([box])], [make_result([])]]

    summary = run_classification(_config(src, dst))

    assert summary.with_person == 1
    assert summary.copied == 1  # non-person not copied in extract mode
    assert (dst / "a.png").exists()
    assert not (dst / "b.png").exists()


def test_corrupt_image_is_skipped(fake_yolo, tmp_path: Path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    write_corrupt_image(src / "bad.jpg")

    fake_yolo.return_value.predict.return_value = [make_result([])]
    summary = run_classification(_config(src, dst))

    assert summary.corrupt == 1
    assert summary.copied == 0


def test_missing_source_raises_pipeline_error(fake_yolo, tmp_path: Path):
    with pytest.raises(PipelineError):
        run_classification(_config(tmp_path / "nope", tmp_path / "dst"))


def test_empty_source_returns_zero(fake_yolo, tmp_path: Path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    summary = run_classification(_config(src, dst))
    assert summary.total == 0
    assert summary.copied == 0


def test_stop_callback_halts(fake_yolo, tmp_path: Path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    for i in range(5):
        write_valid_image(src / f"img_{i}.png")
    fake_yolo.return_value.predict.return_value = [make_result([])]

    calls = {"n": 0}

    def should_stop() -> bool:
        calls["n"] += 1
        return calls["n"] > 2

    summary = run_classification(_config(src, dst), should_stop=should_stop)
    assert summary.stopped is True
