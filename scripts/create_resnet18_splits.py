"""
Create supervised ResNet18 splits for all MVTec categories.

Purpose
-------
ResNet18 benchmark trains one category-specific ResNet18 model per MVTec category.

Why category-specific?
----------------------
This gives an apples-to-apples comparison against category-specific Anomalib
models. Each object category has a different normal appearance, so one shared
CNN can learn category shortcuts instead of defect behavior.

Split strategy
--------------
MVTec AD has:
- train/good: normal images
- test/good: normal test images
- test/<defect_type>: defective test images

For supervised CNN training, only a limited fraction of
defective test images as labeled defect examples. This simulates real
manufacturing where defect labels are scarce.

For each category:
- train split:
    normal train/good subset
    limited defective samples
- validation split:
    carved from the train split for early stopping
- eval split:
    test/good + remaining defective samples

Important:
- Eval defective images are not used in training.
- Validation images are used for early stopping only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from defect_detection.benchmark.splits import create_category_splits  # noqa: E402
from defect_detection.utils.config import load_yaml_config  # noqa: E402
from defect_detection.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def main() -> int:
    """Create ResNet18 benchmark splits."""
    parser = argparse.ArgumentParser(description="Create ResNet18 splits.")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Optional category subset. Defaults to all categories in config.",
    )
    args = parser.parse_args()

    config = load_yaml_config(PROJECT_ROOT / "configs/experiments/resnet18_benchmark.yaml")

    metadata_path = PROJECT_ROOT / config["dataset"]["metadata_csv"]
    output_dir = PROJECT_ROOT / config["outputs"]["split_dir"]

    output_dir.mkdir(parents=True, exist_ok=True)

    categories = args.categories or config["dataset"]["categories"]
    metadata_df = pd.read_csv(metadata_path)

    split_summary_rows = []

    for category in categories:
        logger.info("Creating ResNet18 benchmark split for category: %s", category)

        train_df, validation_df, eval_df = create_category_splits(
            metadata_df=metadata_df,
            category=category,
            normal_train_fraction=float(config["split"]["normal_train_fraction"]),
            defect_train_fraction=float(config["split"]["defect_train_fraction"]),
            validation_fraction=float(config["split"]["validation_fraction"]),
            random_seed=int(config["split"]["random_seed"]),
        )

        category_dir = output_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)

        train_df.to_csv(category_dir / "train.csv", index=False)
        validation_df.to_csv(category_dir / "validation.csv", index=False)
        eval_df.to_csv(category_dir / "eval.csv", index=False)

        split_summary_rows.append(
            {
                "category": category,
                "train_total": len(train_df),
                "train_normal": int((train_df["label"] == "normal").sum()),
                "train_defective": int((train_df["label"] == "defective").sum()),
                "validation_total": len(validation_df),
                "validation_normal": int((validation_df["label"] == "normal").sum()),
                "validation_defective": int((validation_df["label"] == "defective").sum()),
                "eval_total": len(eval_df),
                "eval_normal": int((eval_df["label"] == "normal").sum()),
                "eval_defective": int((eval_df["label"] == "defective").sum()),
            }
        )

        logger.info(
            "%s | train=%d | validation=%d | eval=%d",
            category,
            len(train_df),
            len(validation_df),
            len(eval_df),
        )

    summary_df = pd.DataFrame(split_summary_rows)
    summary_df.to_csv(output_dir / "split_summary.csv", index=False)

    logger.info("Saved split summary: %s", output_dir / "split_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
