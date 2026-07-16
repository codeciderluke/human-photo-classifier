"""Unit tests for ``inspect_image`` corruption-level detection."""

from __future__ import annotations

from pathlib import Path

import pytest

from filters.image_integrity import CORRUPT, OK, PARTIAL, inspect_image

# This check requires Pillow.
pytest.importorskip("PIL")


def _make_ok_jpeg(path: Path) -> Path:
    from PIL import Image

    Image.new("RGB", (240, 240), (120, 80, 40)).save(path, quality=95)
    return path


def test_ok_image(tmp_path: Path):
    path = _make_ok_jpeg(tmp_path / "ok.jpg")
    assert inspect_image(path) == OK


def test_corrupt_image(tmp_path: Path):
    path = tmp_path / "corrupt.jpg"
    path.write_bytes(b"this is not an image at all")
    assert inspect_image(path) == CORRUPT


def test_partial_image_truncated(tmp_path: Path):
    # Make a valid JPEG, then truncate the tail to create partial corruption.
    ok = _make_ok_jpeg(tmp_path / "src.jpg")
    data = ok.read_bytes()
    partial = tmp_path / "partial.jpg"
    partial.write_bytes(data[: int(len(data) * 0.6)])
    assert inspect_image(partial) == PARTIAL


def test_missing_pil_returns_ok(tmp_path, monkeypatch):
    # When Pillow import fails, skip the check and return OK.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("PIL"):
            raise ImportError("no PIL")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert inspect_image(tmp_path / "whatever.jpg") == OK
