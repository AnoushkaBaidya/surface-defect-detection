"""
FastAPI request and response schemas.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ClassificationRequest(BaseModel):
    image_path: str = Field(..., description="Path to an image file on disk.")
    model_name: str = Field(
        default="resnet18_binary_classifier",
        description="Registered runtime model name.",
    )
    checkpoint_path: str | None = Field(
        default=None,
        description="Optional path to a model checkpoint or state_dict file.",
    )
    pretrained: bool = Field(default=False, description="Whether to initialize ImageNet weights.")
    device: str = Field(default="cpu", description="Inference device, for example 'cpu' or 'cuda'.")
    image_size: int = Field(default=224, ge=32, le=1024)


class ClassificationResponse(BaseModel):
    model_name: str
    predicted_label: str
    predicted_index: int
    defect_probability: float
    normal_probability: float
    image_path: str


class HealthResponse(BaseModel):
    status: str
