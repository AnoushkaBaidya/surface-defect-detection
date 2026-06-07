from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from drift.schemas import DriftProfile, FeatureSummary, ImageFeatureRow

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
IMAGE_QUALITY_FEATURES = [
    "brightness_mean",
    "contrast_std",
    "sharpness_laplacian_var",
]


def list_images(folder: Path, label: int) -> list[tuple[Path, int]]:
    """Return labeled image paths from a folder without failing on missing folders."""
    if not folder.exists():
        return []

    return [
        (path, label)
        for path in sorted(folder.rglob("*"))
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def extract_image_features(image_path: Path, label: int) -> ImageFeatureRow:
    """Extract simple image-quality features used for production drift monitoring."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    return ImageFeatureRow(
        image_path=str(image_path),
        label=int(label),
        brightness_mean=float(np.mean(image)),
        contrast_std=float(np.std(image)),
        sharpness_laplacian_var=float(cv2.Laplacian(image, cv2.CV_64F).var()),
    )


def build_feature_rows(normal_dir: Path, defective_dir: Path) -> list[ImageFeatureRow]:
    """Build feature rows from normal and defective folders."""
    image_label_pairs = list_images(normal_dir, label=0) + list_images(defective_dir, label=1)
    return [extract_image_features(path, label) for path, label in image_label_pairs]


def summarize_values(values: np.ndarray) -> FeatureSummary:
    if values.size == 0:
        return FeatureSummary(mean=0.0, std=0.0, p05=0.0, p50=0.0, p95=0.0)

    return FeatureSummary(
        mean=float(np.mean(values)),
        std=float(np.std(values)),
        p05=float(np.percentile(values, 5)),
        p50=float(np.percentile(values, 50)),
        p95=float(np.percentile(values, 95)),
    )


def summarize_profile(rows: list[ImageFeatureRow], profile_name: str) -> DriftProfile:
    labels = [row.label for row in rows]

    feature_summaries = {
        feature: summarize_values(np.array([getattr(row, feature) for row in rows], dtype=float))
        for feature in IMAGE_QUALITY_FEATURES
    }

    return DriftProfile(
        profile_name=profile_name,
        image_count=len(rows),
        defect_rate=float(sum(labels) / len(labels)) if labels else 0.0,
        features=feature_summaries,
        rows=rows,
    )
