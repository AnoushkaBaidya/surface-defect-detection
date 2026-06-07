from __future__ import annotations

import pandas as pd
import pytest

from defect_detection.benchmark.splits import create_category_splits, validate_split_integrity


def build_metadata() -> pd.DataFrame:
    rows = []
    for idx in range(6):
        rows.append(
            {
                "category": "bottle",
                "split": "train",
                "label": "normal",
                "defect_type": "good",
                "image_path": f"train_{idx}.png",
            }
        )
    for idx in range(4):
        rows.append(
            {
                "category": "bottle",
                "split": "test",
                "label": "normal",
                "defect_type": "good",
                "image_path": f"test_good_{idx}.png",
            }
        )
    for idx in range(6):
        rows.append(
            {
                "category": "bottle",
                "split": "test",
                "label": "defective",
                "defect_type": "scratch",
                "image_path": f"test_bad_{idx}.png",
            }
        )
    return pd.DataFrame(rows)


def test_create_category_splits_produces_non_overlapping_partitions() -> None:
    train_df, validation_df, eval_df = create_category_splits(
        metadata_df=build_metadata(),
        category="bottle",
        normal_train_fraction=1.0,
        defect_train_fraction=0.5,
        validation_fraction=0.2,
        random_seed=42,
    )

    assert set(train_df["benchmark_split"]) == {"train"}
    assert set(validation_df["benchmark_split"]) == {"validation"}
    assert set(eval_df["benchmark_split"]) == {"eval"}
    validate_split_integrity(train_df, validation_df, eval_df)


def test_validate_split_integrity_raises_on_overlap() -> None:
    train_df = pd.DataFrame({"image_path": ["a.png"]})
    validation_df = pd.DataFrame({"image_path": ["a.png"]})
    eval_df = pd.DataFrame({"image_path": ["b.png"]})

    with pytest.raises(ValueError):
        validate_split_integrity(train_df, validation_df, eval_df)
