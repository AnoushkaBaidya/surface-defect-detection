"""
Unit tests for MVTec profiling utilities.

These tests do not require the full MVTec dataset. They validate small utility
behavior so CI remains lightweight and fast.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from defect_detection.data.mvtec_profiler import find_images, get_image_size


def test_find_images_returns_supported_files(tmp_path: Path) -> None:
    """
    Verify find_images only returns supported image extensions.
    """
    image_path = tmp_path / "sample.png"
    text_path = tmp_path / "notes.txt"

    image_path.write_bytes(b"fake")
    text_path.write_text("not an image", encoding="utf-8")

    results = find_images(tmp_path, image_extensions=[".png"])

    assert results == [image_path]


def test_get_image_size_reads_valid_image(tmp_path: Path) -> None:
    """
    Verify get_image_size returns width and height for a valid image.
    """
    image_path = tmp_path / "sample.png"
    image = Image.new("RGB", (128, 64))
    image.save(image_path)

    assert get_image_size(image_path) == (128, 64)


def test_get_image_size_returns_none_for_invalid_image(tmp_path: Path) -> None:
    """
    Verify invalid image files do not crash the profiling pipeline.
    """
    invalid_image_path = tmp_path / "broken.png"
    invalid_image_path.write_text("not really an image", encoding="utf-8")

    assert get_image_size(invalid_image_path) is None
