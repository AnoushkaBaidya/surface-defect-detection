"""
Profile selected MVTec AD categories.

Purpose
-------
This script creates dataset profiling dataset deliverables:

- data/reports/mvtec_dataset_profile.csv
- data/reports/mvtec_image_metadata.csv
- data/reports/mvtec_dataset_profile_summary.json
- data/reports/mvtec_dataset_report.md
- artifacts/reports/<category>_normal_samples.png
- artifacts/reports/<category>_defect_samples.png
- artifacts/reports/<category>_mask_samples.png

Run:

    python scripts/profile_dataset.py

Optional:

    python scripts/profile_dataset.py --categories bottle
    python scripts/profile_dataset.py --categories bottle screw
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from defect_detection.data.mvtec_profiler import (  # noqa: E402
    category_profiles_to_dataframe,
    create_report_markdown,
    create_sample_grid,
    image_records_to_dataframe,
    profile_category,
    save_json,
)
from defect_detection.utils.config import load_yaml_config  # noqa: E402
from defect_detection.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def create_category_sample_grids(
    category: str,
    image_metadata_df,
    project_root: Path,
    artifacts_report_dir: Path,
    max_images: int,
    thumbnail_size: int,
) -> None:
    """
    Create normal, defective, and mask sample grids for one category.

    Parameters
    ----------
    category:
        MVTec category name.
    image_metadata_df:
        Image-level metadata dataframe.
    project_root:
        Project root directory.
    artifacts_report_dir:
        Directory where sample grids are saved.
    max_images:
        Maximum images per grid.
    thumbnail_size:
        Thumbnail size used for display.
    """
    category_df = image_metadata_df[image_metadata_df["category"] == category]

    train_normal_paths = [
        project_root / path
        for path in category_df[
            (category_df["split"] == "train") & (category_df["label"] == "normal")
        ]["image_path"].tolist()
    ]

    defect_paths = [
        project_root / path
        for path in category_df[
            (category_df["split"] == "test") & (category_df["label"] == "defective")
        ]["image_path"].tolist()
    ]

    mask_paths = [
        project_root / path
        for path in category_df[
            (category_df["split"] == "test")
            & (category_df["label"] == "defective")
            & (category_df["mask_path"] != "")
        ]["mask_path"].tolist()
    ]

    create_sample_grid(
        train_normal_paths,
        artifacts_report_dir / f"{category}_normal_samples.png",
        title=f"{category} - Normal Training Samples",
        max_images=max_images,
        thumbnail_size=thumbnail_size,
    )

    create_sample_grid(
        defect_paths,
        artifacts_report_dir / f"{category}_defect_samples.png",
        title=f"{category} - Defective Test Samples",
        max_images=max_images,
        thumbnail_size=thumbnail_size,
    )

    create_sample_grid(
        mask_paths,
        artifacts_report_dir / f"{category}_mask_samples.png",
        title=f"{category} - Ground Truth Mask Samples",
        max_images=max_images,
        thumbnail_size=thumbnail_size,
    )


def main() -> int:
    """
    CLI entry point.

    Returns
    -------
    int
        Exit code. 0 means success.
    """
    parser = argparse.ArgumentParser(description="Profile selected MVTec AD categories.")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Categories to profile. Defaults to configured selected categories.",
    )

    args = parser.parse_args()

    config = load_yaml_config(PROJECT_ROOT / "configs/data/mvtec.yaml")

    mvtec_root = PROJECT_ROOT / config["dataset"]["root_dir"]
    reports_dir = PROJECT_ROOT / "data/reports"
    artifacts_report_dir = PROJECT_ROOT / "artifacts/reports"

    image_extensions = config["profiling"]["image_extensions"]
    max_images = int(config["profiling"]["sample_grid"]["max_images"])
    thumbnail_size = int(config["profiling"]["sample_grid"]["thumbnail_size"])

    selected_categories = args.categories or config["profiling"]["selected_categories"]

    logger.info("Starting MVTec AD profiling.")
    logger.info("Dataset root: %s", mvtec_root)
    logger.info("Selected categories: %s", selected_categories)

    if not mvtec_root.exists():
        logger.error("Dataset root not found: %s", mvtec_root)
        logger.error("Run: python scripts/download_mvtec.py")
        return 1

    reports_dir.mkdir(parents=True, exist_ok=True)
    artifacts_report_dir.mkdir(parents=True, exist_ok=True)

    category_profiles = []
    all_image_records = []

    for category in selected_categories:
        logger.info("Profiling category: %s", category)
        profile, records = profile_category(
            mvtec_root=mvtec_root,
            category=category,
            image_extensions=image_extensions,
        )

        category_profiles.append(profile)
        all_image_records.extend(records)

    profiles_df = category_profiles_to_dataframe(category_profiles)
    image_metadata_df = image_records_to_dataframe(all_image_records, PROJECT_ROOT)

    profile_csv_path = reports_dir / "mvtec_dataset_profile.csv"
    image_metadata_csv_path = reports_dir / "mvtec_image_metadata.csv"
    summary_json_path = reports_dir / "mvtec_dataset_profile_summary.json"
    report_md_path = reports_dir / "mvtec_dataset_report.md"

    profiles_df.to_csv(profile_csv_path, index=False)
    image_metadata_df.to_csv(image_metadata_csv_path, index=False)

    summary = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset_name": config["dataset"]["name"],
        "dataset_root": str(mvtec_root.relative_to(PROJECT_ROOT)),
        "selected_categories": selected_categories,
        "total_categories_profiled": len(selected_categories),
        "total_images": int(len(image_metadata_df)),
        "total_train_images": (
            int((image_metadata_df["split"] == "train").sum()) if len(image_metadata_df) else 0
        ),
        "total_test_images": (
            int((image_metadata_df["split"] == "test").sum()) if len(image_metadata_df) else 0
        ),
        "total_defective_test_images": (
            int((image_metadata_df["label"] == "defective").sum()) if len(image_metadata_df) else 0
        ),
        "category_profiles": profiles_df.to_dict(orient="records"),
    }

    save_json(summary, summary_json_path)

    report_markdown = create_report_markdown(
        profiles_df=profiles_df,
        image_records_df=image_metadata_df,
        selected_categories=selected_categories,
    )
    report_md_path.write_text(report_markdown, encoding="utf-8")

    for category in selected_categories:
        create_category_sample_grids(
            category=category,
            image_metadata_df=image_metadata_df,
            project_root=PROJECT_ROOT,
            artifacts_report_dir=artifacts_report_dir,
            max_images=max_images,
            thumbnail_size=thumbnail_size,
        )

    logger.info("Saved category profile: %s", profile_csv_path)
    logger.info("Saved image metadata: %s", image_metadata_csv_path)
    logger.info("Saved summary JSON: %s", summary_json_path)
    logger.info("Saved markdown report: %s", report_md_path)
    logger.info("Saved sample grids under: %s", artifacts_report_dir)

    invalid_categories = profiles_df[profiles_df["is_valid_for_anomalib"] != True]  # noqa: E712
    if not invalid_categories.empty:
        logger.error("Some categories are not valid for Anomalib:")
        logger.error("%s", invalid_categories[["category", "notes"]].to_string(index=False))
        return 1

    logger.info("dataset profiling profiling completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
