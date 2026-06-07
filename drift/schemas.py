from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ImageFeatureRow:
    image_path: str
    label: int
    brightness_mean: float
    contrast_std: float
    sharpness_laplacian_var: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FeatureSummary:
    mean: float
    std: float
    p05: float
    p50: float
    p95: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class DriftProfile:
    profile_name: str
    image_count: int
    defect_rate: float
    features: dict[str, FeatureSummary]
    rows: list[ImageFeatureRow]

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "image_count": self.image_count,
            "defect_rate": self.defect_rate,
            "features": {feature: summary.to_dict() for feature, summary in self.features.items()},
            "rows": [row.to_dict() for row in self.rows],
        }
