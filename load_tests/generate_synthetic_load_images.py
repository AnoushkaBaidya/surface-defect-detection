from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def list_images(folder: Path) -> list[Path]:
    return sorted(path for path in folder.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def augment(image: Image.Image, rng: random.Random) -> Image.Image:
    image = image.convert("RGB")

    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.75, 1.20))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.75, 1.20))
    image = ImageEnhance.Sharpness(image).enhance(rng.uniform(0.75, 1.25))

    if rng.random() < 0.30:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 0.8)))

    return image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    source_images = list_images(source_dir)

    if not source_images:
        raise FileNotFoundError(f"No images found in {source_dir}")

    for index in range(args.count):
        source_path = rng.choice(source_images)

        with Image.open(source_path) as image:
            output = augment(image, rng)

        output_path = output_dir / f"synthetic_load_{index:06d}.jpg"
        output.save(output_path, quality=90)

        if (index + 1) % 100 == 0:
            print(f"Created {index + 1}/{args.count}")

    print(f"Done. Images saved to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
