from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_image(image_bytes: bytes) -> Image.Image:
    """Load uploaded bytes as RGB PIL image."""
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def validate_image_dimensions(image: Image.Image, min_dimension: int, max_dimension: int) -> None:
    width, height = image.size

    if width < min_dimension or height < min_dimension:
        raise ValueError(
            "Image is too small for inference: "
            f"width={width}, height={height}, minimum={min_dimension}"
        )

    if width > max_dimension or height > max_dimension:
        raise ValueError(
            "Image is too large for inference: "
            f"width={width}, height={height}, maximum={max_dimension}"
        )


def preprocess_pil_for_resnet(image: Image.Image, input_size: int) -> np.ndarray:
    """
    Convert PIL image into ONNX Runtime tensor.

    Output shape:
        [1, 3, H, W]
    """
    resized = image.resize((input_size, input_size))

    array = np.asarray(resized).astype(np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    array = np.transpose(array, (2, 0, 1))
    array = np.expand_dims(array, axis=0)

    return np.ascontiguousarray(array, dtype=np.float32)


def preprocess_pil_for_efficientad(image: Image.Image, input_size: int) -> np.ndarray:
    """
    Convert PIL image into EfficientAD ONNX Runtime input tensor.

    EfficientAD was trained using torchvision ToTensor without ImageNet mean/std.
    Output shape:
        [1, 3, H, W]
    """
    resized = image.resize((input_size, input_size))

    array = np.asarray(resized).astype(np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    array = np.expand_dims(array, axis=0)

    return np.ascontiguousarray(array, dtype=np.float32)
