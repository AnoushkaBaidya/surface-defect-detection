"""
PyTorch dataset utilities for supervised defect classification.

Why this exists
---------------
The CNN baseline is intentionally different from anomaly detection.

Anomaly detection:
- trains mostly on normal/good images
- detects deviations from normal appearance

CNN supervised baseline:
- needs normal and defective labels during training
- struggles when defect labels are rare
- helps demonstrate why anomaly detection is preferred in this benchmark

This file provides a clean Dataset class that reads from the dataset profiling
image-level metadata CSV.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MVTecClassificationDataset(Dataset):
    """
    Dataset for binary normal/defective classification.

    Label mapping:
    - normal -> 0
    - defective -> 1

    Parameters
    ----------
    dataframe:
        DataFrame containing image paths and labels.
    project_root:
        Repository root path.
    image_size:
        Resize dimension for CNN input.
    is_train:
        Whether to apply training augmentations.
    """

    LABEL_TO_INDEX = {
        "normal": 0,
        "defective": 1,
    }

    def __init__(
        self,
        dataframe: pd.DataFrame,
        project_root: Path,
        image_size: int,
        is_train: bool,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.project_root = project_root
        self.image_size = image_size
        self.is_train = is_train
        self.transform = self._build_transform()

    def _build_transform(self) -> transforms.Compose:
        """
        Build image transformations.

        Training transforms are intentionally light because industrial inspection
        should not distort defects too aggressively.
        """
        if self.is_train:
            return transforms.Compose(
                [
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=5),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )

        return transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.dataframe)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load one image and label.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            image tensor and integer label tensor.
        """
        row = self.dataframe.iloc[index]

        image_path = self.project_root / row["image_path"]
        label_text = row["label"]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        image_tensor = self.transform(image)
        label_tensor = torch.tensor(self.LABEL_TO_INDEX[label_text], dtype=torch.long)

        return image_tensor, label_tensor
