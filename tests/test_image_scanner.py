"""Unit tests for ``ImageScanner`` and ``FileCopyService``."""

from __future__ import annotations

from pathlib import Path

import pytest

from services import FileCopyService, ImageScanner


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"data")
    return path


# ---------------------------------------------------------------- scanner

def test_scan_collects_only_image_extensions(tmp_path: Path):
    _touch(tmp_path / "a.jpg")
    _touch(tmp_path / "b.PNG")
    _touch(tmp_path / "c.txt")
    _touch(tmp_path / "d.doc")

    scanner = ImageScanner()
    found = scanner.scan(tmp_path)

    names = sorted(p.name for p in found)
    assert names == ["a.jpg", "b.PNG"]


def test_scan_recursive_includes_subfolders(tmp_path: Path):
    _touch(tmp_path / "top.jpg")
    _touch(tmp_path / "sub" / "nested.jpg")

    scanner = ImageScanner(recursive=True)
    found = {p.name for p in scanner.scan(tmp_path)}
    assert found == {"top.jpg", "nested.jpg"}


def test_scan_non_recursive_excludes_subfolders(tmp_path: Path):
    _touch(tmp_path / "top.jpg")
    _touch(tmp_path / "sub" / "nested.jpg")

    scanner = ImageScanner(recursive=False)
    found = {p.name for p in scanner.scan(tmp_path)}
    assert found == {"top.jpg"}


def test_scan_missing_folder_raises(tmp_path: Path):
    scanner = ImageScanner()
    with pytest.raises(FileNotFoundError):
        scanner.scan(tmp_path / "nope")


# ------------------------------------------------------------- copy service

def test_copy_preserves_structure(tmp_path: Path):
    source_root = tmp_path / "src"
    dest_root = tmp_path / "dst"
    image = _touch(source_root / "2024" / "photo.jpg")

    service = FileCopyService(source_root, dest_root)
    target = service.copy(image)

    assert target == dest_root / "2024" / "photo.jpg"
    assert target.exists()


def test_copy_deduplicates_name(tmp_path: Path):
    source_root = tmp_path / "src"
    dest_root = tmp_path / "dst"
    image = _touch(source_root / "photo.jpg")

    service = FileCopyService(source_root, dest_root)
    first = service.copy(image)
    second = service.copy(image)

    assert first.name == "photo.jpg"
    assert second.name == "photo_1.jpg"
    assert first.exists() and second.exists()


def test_copy_flat_when_structure_disabled(tmp_path: Path):
    source_root = tmp_path / "src"
    dest_root = tmp_path / "dst"
    image = _touch(source_root / "deep" / "photo.jpg")

    service = FileCopyService(source_root, dest_root, preserve_structure=False)
    target = service.copy(image)

    assert target == dest_root / "photo.jpg"


def test_copy_into_category_subfolder(tmp_path: Path):
    source_root = tmp_path / "src"
    dest_root = tmp_path / "dst"
    image = _touch(source_root / "2024" / "photo.jpg")

    service = FileCopyService(source_root, dest_root)
    target = service.copy(image, subfolder="with_face/male")

    assert target == dest_root / "with_face" / "male" / "2024" / "photo.jpg"
    assert target.exists()
