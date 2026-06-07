"""
Helpers for benchmark split construction and validation.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def safe_train_test_split(
    dataframe: pd.DataFrame,
    train_size: float,
    random_seed: int,
    stratify_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataframe with safe fallback for tiny classes."""
    if len(dataframe) < 2:
        return dataframe.copy(), dataframe.iloc[0:0].copy()

    stratify_values = None
    if stratify_column is not None and stratify_column in dataframe.columns:
        value_counts = dataframe[stratify_column].value_counts()
        if not value_counts.empty and value_counts.min() >= 2:
            stratify_values = dataframe[stratify_column]

    try:
        return train_test_split(
            dataframe,
            train_size=train_size,
            random_state=random_seed,
            shuffle=True,
            stratify=stratify_values,
        )
    except ValueError:
        return train_test_split(
            dataframe,
            train_size=train_size,
            random_state=random_seed,
            shuffle=True,
            stratify=None,
        )


def create_category_splits(
    metadata_df: pd.DataFrame,
    category: str,
    normal_train_fraction: float,
    defect_train_fraction: float,
    validation_fraction: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train/validation/eval splits for one MVTec category."""
    category_df = metadata_df[metadata_df["category"] == category].copy()

    normal_train_pool = category_df[
        (category_df["split"] == "train") & (category_df["label"] == "normal")
    ].copy()
    test_good_df = category_df[
        (category_df["split"] == "test") & (category_df["label"] == "normal")
    ].copy()
    defective_df = category_df[
        (category_df["split"] == "test") & (category_df["label"] == "defective")
    ].copy()

    if normal_train_pool.empty:
        raise ValueError(f"No normal train/good images found for category: {category}")
    if defective_df.empty:
        raise ValueError(f"No defective test images found for category: {category}")

    if normal_train_fraction < 1.0:
        normal_train_df, _ = safe_train_test_split(
            dataframe=normal_train_pool,
            train_size=normal_train_fraction,
            random_seed=random_seed,
        )
    else:
        normal_train_df = normal_train_pool.copy()

    defect_train_df, defect_eval_df = safe_train_test_split(
        dataframe=defective_df,
        train_size=defect_train_fraction,
        random_seed=random_seed,
        stratify_column="defect_type",
    )

    supervised_train_pool = pd.concat([normal_train_df, defect_train_df], ignore_index=True)
    supervised_train_pool = supervised_train_pool.sample(
        frac=1.0,
        random_state=random_seed,
    ).reset_index(drop=True)

    train_df, validation_df = safe_train_test_split(
        dataframe=supervised_train_pool,
        train_size=1.0 - validation_fraction,
        random_seed=random_seed,
        stratify_column="label",
    )

    eval_df = pd.concat([test_good_df, defect_eval_df], ignore_index=True)
    eval_df = eval_df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    train_df["benchmark_split"] = "train"
    validation_df["benchmark_split"] = "validation"
    eval_df["benchmark_split"] = "eval"

    validate_split_integrity(train_df, validation_df, eval_df)
    return train_df, validation_df, eval_df


def validate_split_integrity(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    eval_df: pd.DataFrame,
) -> None:
    """Validate that split files do not overlap on image_path."""
    train_paths = set(train_df["image_path"].tolist())
    validation_paths = set(validation_df["image_path"].tolist())
    eval_paths = set(eval_df["image_path"].tolist())

    if train_paths & validation_paths:
        raise ValueError("Train and validation splits overlap.")
    if train_paths & eval_paths:
        raise ValueError("Train and eval splits overlap.")
    if validation_paths & eval_paths:
        raise ValueError("Validation and eval splits overlap.")
