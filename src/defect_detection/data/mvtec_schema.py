"""
Schema definitions for the MVTec AD dataset.

Why this file exists
--------------------
The dataset code should not rely on loose dictionaries everywhere.
Small typed data containers make the dataset profiling and validation code
easier to understand, test, and extend.

MVTec AD structure expected by this project
------------------------------------------
data/raw/mvtec_ad/<category>/
    train/
        good/
            image.png
    test/
        good/
            image.png
        <defect_type>/
            image.png
    ground_truth/
        <defect_type>/
            image_mask.png

Important:
- Training images are expected to be normal/good only.
- Test images include both good and defective samples.
- Ground-truth masks exist for defective test images.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CategoryProfile:
    """
    Summary statistics for one MVTec AD category.

    Attributes
    ----------
    category:
        MVTec category name, for example 'bottle' or 'screw'.
    category_dir:
        Absolute path to the category directory.
    train_good_count:
        Number of normal training images.
    test_good_count:
        Number of normal test images.
    test_defect_count:
        Number of defective test images across all defect types.
    ground_truth_mask_count:
        Number of available pixel-level mask images.
    defect_types:
        Defect type folder names in test/.
    image_width_min:
        Minimum image width found in the category.
    image_width_max:
        Maximum image width found in the category.
    image_height_min:
        Minimum image height found in the category.
    image_height_max:
        Maximum image height found in the category.
    is_valid_for_anomalib:
        Whether required train/test folders exist for Anomalib-style loading.
    notes:
        Human-readable validation notes.
    """

    category: str
    category_dir: Path
    train_good_count: int
    test_good_count: int
    test_defect_count: int
    ground_truth_mask_count: int
    defect_types: list[str]
    image_width_min: int | None
    image_width_max: int | None
    image_height_min: int | None
    image_height_max: int | None
    is_valid_for_anomalib: bool
    notes: list[str]


@dataclass(frozen=True)
class ImageRecord:
    """
    Metadata for a single image.

    This is useful for building CSV/Parquet metadata for
    CNN baselines, low-data splits, and related comparisons.
    """

    category: str
    split: str
    label: str
    defect_type: str
    image_path: Path
    mask_path: Path | None
    width: int
    height: int
