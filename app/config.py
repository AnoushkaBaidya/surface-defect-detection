from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[1]


class ResNetConfig(BaseModel):
    onnx_path: str
    input_size: int
    raw_threshold: float


class EfficientADConfig(BaseModel):
    enabled: bool
    threshold: float
    runtime: str = "onnx"
    onnx_path: str | None = None
    calibration_path: str | None = None
    input_size: int = 224
    note: str | None = None


class ModelPolicyConfig(BaseModel):
    name: str
    version: str


class ApiLimitsConfig(BaseModel):
    max_upload_bytes: int = 10 * 1024 * 1024
    min_image_dimension: int = 32
    max_image_dimension: int = 4096


class AppConfig(BaseModel):
    model_policy: ModelPolicyConfig
    resnet18: ResNetConfig
    efficientad: EfficientADConfig
    api_limits: ApiLimitsConfig


def resolve_repo_path(relative_path: str) -> Path:
    return REPO_ROOT / relative_path


def config_path_from_env() -> Path:
    config_path = os.getenv("SURFACE_DEFECT_CONFIG_PATH", "configs/final_model_policy.yaml")
    return resolve_repo_path(config_path)


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw is not None else default


def load_app_config() -> AppConfig:
    config_path = config_path_from_env()

    with config_path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)

    payload.setdefault("api_limits", {})
    payload["api_limits"]["max_upload_bytes"] = env_int(
        "SURFACE_DEFECT_MAX_UPLOAD_BYTES",
        int(payload["api_limits"].get("max_upload_bytes", 10 * 1024 * 1024)),
    )
    payload["api_limits"]["min_image_dimension"] = env_int(
        "SURFACE_DEFECT_MIN_IMAGE_DIMENSION",
        int(payload["api_limits"].get("min_image_dimension", 32)),
    )
    payload["api_limits"]["max_image_dimension"] = env_int(
        "SURFACE_DEFECT_MAX_IMAGE_DIMENSION",
        int(payload["api_limits"].get("max_image_dimension", 4096)),
    )

    return AppConfig(**payload)
