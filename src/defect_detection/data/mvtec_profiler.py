"""
MVTec AD dataset profiling utilities.

Purpose
-------
This module validates and profiles the MVTec AD directory structure.

It creates:
- category-level summary statistics
- image-level metadata
- sample grids for README/reporting
- markdown report content

Design principles
-----------------
1. Keep data inspection reproducible.
2. Save machine-readable outputs such as CSV/JSON.
3. Save human-readable outputs such as PNG grids and markdown.
4. Avoid hardcoding category names inside the logic.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, UnidentifiedImageError

from defect_detection.data.mvtec_schema import CategoryProfile, ImageRecord
from defect_detection.utils.logging import get_logger

logger = get_logger(__name__)


def find_images(directory: Path, image_extensions: Iterable[str]) -> list[Path]:
    """
    Recursively find image files in a directory.

    Parameters
    ----------
    directory:
        Directory to search.
    image_extensions:
        Supported image extensions, for example ['.png', '.jpg'].

    Returns
    -------
    list[Path]
        Sorted list of image paths.
    """
    if not directory.exists():
        return []

    normalized_extensions = {ext.lower() for ext in image_extensions}

    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in normalized_extensions
    )


def get_image_size(image_path: Path) -> tuple[int, int] | None:
    """
    Read image width and height safely.

    Parameters
    ----------
    image_path:
        Path to image file.

    Returns
    -------
    tuple[int, int] | None
        (width, height), or None if the file cannot be opened.
    """
    try:
        with Image.open(image_path) as image:
            return image.size
    except (OSError, UnidentifiedImageError) as error:
        logger.warning("Could not read image %s: %s", image_path, error)
        return None


def infer_mask_path(category_dir: Path, defect_type: str, image_path: Path) -> Path | None:
    """
    Infer the ground-truth mask path for a defective test image.

    MVTec masks usually follow this pattern:
    ground_truth/<defect_type>/<image_stem>_mask.png

    Parameters
    ----------
    category_dir:
        Category root directory.
    defect_type:
        Defect type folder name.
    image_path:
        Defective image path.

    Returns
    -------
    Path | None
        Mask path if it exists, otherwise None.
    """
    mask_path = category_dir / "ground_truth" / defect_type / f"{image_path.stem}_mask.png"
    return mask_path if mask_path.exists() else None


def build_image_records(
    category_dir: Path,
    category: str,
    image_extensions: Iterable[str],
) -> list[ImageRecord]:
    """
    Build image-level metadata records for one category.

    Parameters
    ----------
    category_dir:
        Path to one MVTec category directory.
    category:
        Category name.
    image_extensions:
        Supported image extensions.

    Returns
    -------
    list[ImageRecord]
        Image-level metadata records.
    """
    records: list[ImageRecord] = []

    train_good_dir = category_dir / "train" / "good"
    for image_path in find_images(train_good_dir, image_extensions):
        size = get_image_size(image_path)
        if size is None:
            continue

        width, height = size
        records.append(
            ImageRecord(
                category=category,
                split="train",
                label="normal",
                defect_type="good",
                image_path=image_path,
                mask_path=None,
                width=width,
                height=height,
            )
        )

    test_dir = category_dir / "test"
    if test_dir.exists():
        for defect_dir in sorted(path for path in test_dir.iterdir() if path.is_dir()):
            defect_type = defect_dir.name
            label = "normal" if defect_type == "good" else "defective"

            for image_path in find_images(defect_dir, image_extensions):
                size = get_image_size(image_path)
                if size is None:
                    continue

                width, height = size
                mask_path = None
                if label == "defective":
                    mask_path = infer_mask_path(category_dir, defect_type, image_path)

                records.append(
                    ImageRecord(
                        category=category,
                        split="test",
                        label=label,
                        defect_type=defect_type,
                        image_path=image_path,
                        mask_path=mask_path,
                        width=width,
                        height=height,
                    )
                )

    return records


def profile_category(
    mvtec_root: Path,
    category: str,
    image_extensions: Iterable[str],
) -> tuple[CategoryProfile, list[ImageRecord]]:
    """
    Profile one MVTec AD category.

    Parameters
    ----------
    mvtec_root:
        Root directory containing MVTec categories.
    category:
        Category name.
    image_extensions:
        Supported image extensions.

    Returns
    -------
    tuple[CategoryProfile, list[ImageRecord]]
        Category summary and image-level records.
    """
    category_dir = mvtec_root / category
    notes: list[str] = []

    if not category_dir.exists():
        notes.append(f"Category directory does not exist: {category_dir}")
        return (
            CategoryProfile(
                category=category,
                category_dir=category_dir,
                train_good_count=0,
                test_good_count=0,
                test_defect_count=0,
                ground_truth_mask_count=0,
                defect_types=[],
                image_width_min=None,
                image_width_max=None,
                image_height_min=None,
                image_height_max=None,
                is_valid_for_anomalib=False,
                notes=notes,
            ),
            [],
        )

    records = build_image_records(category_dir, category, image_extensions)

    train_good_count = sum(
        1 for record in records if record.split == "train" and record.defect_type == "good"
    )
    test_good_count = sum(
        1 for record in records if record.split == "test" and record.defect_type == "good"
    )
    test_defect_count = sum(
        1 for record in records if record.split == "test" and record.label == "defective"
    )
    ground_truth_mask_count = sum(1 for record in records if record.mask_path is not None)

    defect_types = sorted(
        {
            record.defect_type
            for record in records
            if record.split == "test" and record.defect_type != "good"
        }
    )

    widths = [record.width for record in records]
    heights = [record.height for record in records]

    train_good_dir = category_dir / "train" / "good"
    test_dir = category_dir / "test"

    is_valid_for_anomalib = train_good_dir.exists() and test_dir.exists() and train_good_count > 0

    if not train_good_dir.exists():
        notes.append(f"Missing train/good directory: {train_good_dir}")

    if not test_dir.exists():
        notes.append(f"Missing test directory: {test_dir}")

    if train_good_count == 0:
        notes.append("No train/good images found.")

    if test_defect_count > 0 and ground_truth_mask_count == 0:
        notes.append("Defective test images exist, but no masks were found.")

    if not notes:
        notes.append("Category structure looks valid.")

    profile = CategoryProfile(
        category=category,
        category_dir=category_dir,
        train_good_count=train_good_count,
        test_good_count=test_good_count,
        test_defect_count=test_defect_count,
        ground_truth_mask_count=ground_truth_mask_count,
        defect_types=defect_types,
        image_width_min=min(widths) if widths else None,
        image_width_max=max(widths) if widths else None,
        image_height_min=min(heights) if heights else None,
        image_height_max=max(heights) if heights else None,
        is_valid_for_anomalib=is_valid_for_anomalib,
        notes=notes,
    )

    return profile, records


def category_profiles_to_dataframe(profiles: list[CategoryProfile]) -> pd.DataFrame:
    """
    Convert category profiles to a tabular dataframe.

    Parameters
    ----------
    profiles:
        Category profiles.

    Returns
    -------
    pd.DataFrame
        Category-level profile table.
    """
    rows = []

    for profile in profiles:
        rows.append(
            {
                "category": profile.category,
                "category_dir": str(profile.category_dir),
                "train_good_count": profile.train_good_count,
                "test_good_count": profile.test_good_count,
                "test_defect_count": profile.test_defect_count,
                "ground_truth_mask_count": profile.ground_truth_mask_count,
                "defect_type_count": len(profile.defect_types),
                "defect_types": ", ".join(profile.defect_types),
                "image_width_min": profile.image_width_min,
                "image_width_max": profile.image_width_max,
                "image_height_min": profile.image_height_min,
                "image_height_max": profile.image_height_max,
                "is_valid_for_anomalib": profile.is_valid_for_anomalib,
                "notes": " | ".join(profile.notes),
            }
        )

    return pd.DataFrame(rows)


def image_records_to_dataframe(records: list[ImageRecord], project_root: Path) -> pd.DataFrame:
    """
    Convert image records to a dataframe.

    Paths are stored relative to the project root so the metadata remains
    portable across machines.

    Parameters
    ----------
    records:
        Image records.
    project_root:
        Project root directory.

    Returns
    -------
    pd.DataFrame
        Image-level metadata table.
    """
    rows = []

    for record in records:
        rows.append(
            {
                "category": record.category,
                "split": record.split,
                "label": record.label,
                "defect_type": record.defect_type,
                "image_path": str(record.image_path.relative_to(project_root)),
                "mask_path": (
                    str(record.mask_path.relative_to(project_root))
                    if record.mask_path is not None
                    else ""
                ),
                "width": record.width,
                "height": record.height,
            }
        )

    return pd.DataFrame(rows)


def create_sample_grid(
    image_paths: list[Path],
    output_path: Path,
    title: str,
    max_images: int = 12,
    thumbnail_size: int = 224,
) -> None:
    """
    Create and save an image sample grid.

    Parameters
    ----------
    image_paths:
        Image paths to visualize.
    output_path:
        Output PNG file path.
    title:
        Figure title.
    max_images:
        Maximum images to include.
    thumbnail_size:
        Resize dimension used for display only.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    selected_paths = image_paths[:max_images]

    if not selected_paths:
        logger.warning("No images available for grid: %s", title)
        return

    columns = min(4, len(selected_paths))
    rows = (len(selected_paths) + columns - 1) // columns

    fig, axes = plt.subplots(rows, columns, figsize=(columns * 3, rows * 3))

    if rows == 1 and columns == 1:
        axes_list = [axes]
    elif rows == 1 or columns == 1:
        axes_list = list(axes)
    else:
        axes_list = [axis for row_axes in axes for axis in row_axes]

    for axis in axes_list:
        axis.axis("off")

    for axis, image_path in zip(axes_list, selected_paths, strict=False):
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image.thumbnail((thumbnail_size, thumbnail_size))
            axis.imshow(image)
            axis.set_title(image_path.parent.name, fontsize=8)
            axis.axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_report_markdown(
    profiles_df: pd.DataFrame,
    image_records_df: pd.DataFrame,
    selected_categories: list[str],
) -> str:
    """
    Create a markdown report for dataset profiling.

    Parameters
    ----------
    profiles_df:
        Category-level profile dataframe.
    image_records_df:
        Image-level records dataframe.
    selected_categories:
        Categories included in the report.

    Returns
    -------
    str
        Markdown report content.
    """
    total_images = len(image_records_df)
    total_train = int((image_records_df["split"] == "train").sum()) if total_images else 0
    total_test = int((image_records_df["split"] == "test").sum()) if total_images else 0
    total_defective = int((image_records_df["label"] == "defective").sum()) if total_images else 0
    total_normal = int((image_records_df["label"] == "normal").sum()) if total_images else 0

    markdown = [
        "# MVTec AD Dataset Profile",
        "",
        "## Selected Categories",
        "",
        ", ".join(selected_categories),
        "",
        "## Overall Counts",
        "",
        f"- Total images: {total_images}",
        f"- Train images: {total_train}",
        f"- Test images: {total_test}",
        f"- Normal images: {total_normal}",
        f"- Defective test images: {total_defective}",
        "",
        "## Category Summary",
        "",
        profiles_df.to_markdown(index=False),
        "",
        "## Defect Type Counts",
        "",
    ]

    if total_images:
        defect_counts = (
            image_records_df[image_records_df["split"] == "test"]
            .groupby(["category", "defect_type", "label"])
            .size()
            .reset_index(name="count")
        )
        markdown.append(defect_counts.to_markdown(index=False))
    else:
        markdown.append("No image records found.")

    markdown.extend(
        [
            "",
            "## Notes",
            "",
            "- Training data is expected to contain only normal/good images.",
            "- Defective samples are primarily used during evaluation.",
            "- Ground-truth masks are used for defect localization and pixel-level metrics.",
            "- This report supports benchmark reproduction, model comparison, and dataset sanity checks.",
            "",
        ]
    )

    return "\n".join(markdown)


def save_json(data: dict, output_path: Path) -> None:
    """
    Save a dictionary as pretty JSON.

    Parameters
    ----------
    data:
        JSON-serializable dictionary.
    output_path:
        Destination file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
