from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    import ray
except ImportError:  # pragma: no cover - handled in CLI entrypoint
    ray = None


DEFAULT_IMAGE_SIZE = 256
DEFAULT_PREVIEW_LIMIT = 16


def preprocess_record(record: dict[str, Any], image_size: int = DEFAULT_IMAGE_SIZE) -> dict[str, Any]:
    """Resize and normalize a Ray Data image record."""
    image = np.asarray(record["image"])
    pil_image = Image.fromarray(image).convert("RGB").resize((image_size, image_size))
    processed = np.asarray(pil_image, dtype=np.float32) / 255.0
    updated = dict(record)
    updated["image"] = processed
    updated["image_size"] = image_size
    return updated


def augment_record(record: dict[str, Any], horizontal_flip: bool = True) -> dict[str, Any]:
    """Apply a simple deterministic augmentation for pipeline demonstration."""
    image = np.asarray(record["image"])
    augmented = np.flip(image, axis=1).copy() if horizontal_flip else image
    updated = dict(record)
    updated["image"] = augmented
    updated["augmentation"] = "horizontal_flip" if horizontal_flip else "none"
    return updated


def write_preview_outputs(records: list[dict[str, Any]], output_dir: Path) -> Path:
    """Write a small preview set and manifest from transformed Ray records."""
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "preview_images"
    preview_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        image = np.asarray(record["image"])
        image_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        output_path = preview_dir / f"sample_{index:03d}.png"
        Image.fromarray(image_uint8).save(output_path)
        manifest.append(
            {
                "preview_path": str(output_path),
                "image_size": int(record.get("image_size", image_uint8.shape[0])),
                "augmentation": record.get("augmentation", "none"),
            }
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a lightweight Ray Data image preprocessing pipeline.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing source images.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write preview artifacts.")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE, help="Resize target for images.")
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_PREVIEW_LIMIT,
        help="Number of images to materialize for preview output.",
    )
    parser.add_argument(
        "--disable-augment",
        action="store_true",
        help="Skip the demonstration augmentation stage.",
    )
    return parser


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    image_size: int = DEFAULT_IMAGE_SIZE,
    limit: int = DEFAULT_PREVIEW_LIMIT,
    disable_augment: bool = False,
) -> Path:
    if ray is None:
        raise ImportError("Ray is not installed. Install `ray[data]` from `requirements-ml.txt` to run this pipeline.")

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    ray.init(ignore_reinit_error=True, include_dashboard=False)
    try:
        dataset = ray.data.read_images(str(input_dir))
        dataset = dataset.map(lambda row: preprocess_record(row, image_size=image_size))
        if not disable_augment:
            dataset = dataset.map(augment_record)
        records = dataset.limit(limit).take_all()
    finally:
        ray.shutdown()

    return write_preview_outputs(records, output_dir)


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    manifest_path = run_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
        limit=args.limit,
        disable_augment=args.disable_augment,
    )
    print(f"Ray Data preview manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
