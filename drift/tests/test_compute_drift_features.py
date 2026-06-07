from __future__ import annotations

import cv2
import numpy as np

from drift.compute_drift_features import extract_image_features, summarize_profile


def test_extract_image_features_and_profile_summary(tmp_path) -> None:
    image_path = tmp_path / "sample.png"
    image = np.array(
        [
            [0, 0, 0],
            [128, 128, 128],
            [255, 255, 255],
        ],
        dtype=np.uint8,
    )
    cv2.imwrite(str(image_path), image)

    row = extract_image_features(image_path=image_path, label=1)

    assert row.image_path == str(image_path)
    assert row.label == 1
    assert row.brightness_mean > 0
    assert row.contrast_std > 0
    assert row.sharpness_laplacian_var >= 0

    profile = summarize_profile(rows=[row], profile_name="test_profile").to_dict()

    assert profile["profile_name"] == "test_profile"
    assert profile["image_count"] == 1
    assert profile["defect_rate"] == 1.0
    assert "brightness_mean" in profile["features"]
    assert len(profile["rows"]) == 1
