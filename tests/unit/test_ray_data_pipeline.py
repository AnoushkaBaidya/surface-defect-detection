from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "ray_data_pipeline.py"
MODULE_SPEC = importlib.util.spec_from_file_location("ray_data_pipeline", MODULE_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
ray_data_pipeline = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(ray_data_pipeline)

augment_record = ray_data_pipeline.augment_record
preprocess_record = ray_data_pipeline.preprocess_record
write_preview_outputs = ray_data_pipeline.write_preview_outputs


def test_preprocess_record_resizes_and_normalizes() -> None:
    record = {"image": np.zeros((32, 48, 3), dtype=np.uint8)}
    processed = preprocess_record(record, image_size=64)

    assert processed["image"].shape == (64, 64, 3)
    assert processed["image"].dtype == np.float32
    assert processed["image"].min() >= 0.0
    assert processed["image"].max() <= 1.0
    assert processed["image_size"] == 64


def test_augment_record_flips_horizontally() -> None:
    image = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
    record = {"image": image}

    augmented = augment_record(record)

    assert np.array_equal(augmented["image"], np.flip(image, axis=1))
    assert augmented["augmentation"] == "horizontal_flip"


def test_write_preview_outputs_creates_manifest(tmp_path: Path) -> None:
    records = [
        {
            "image": np.ones((16, 16, 3), dtype=np.float32),
            "image_size": 16,
            "augmentation": "horizontal_flip",
        }
    ]

    manifest_path = write_preview_outputs(records, tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest_path.exists()
    assert len(manifest) == 1
    assert Path(manifest[0]["preview_path"]).exists()
    assert manifest[0]["augmentation"] == "horizontal_flip"
