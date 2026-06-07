from app.config import load_app_config, resolve_repo_path


def test_config_loads() -> None:
    config = load_app_config()

    assert config.model_policy.name
    assert config.resnet18.onnx_path
    assert config.efficientad.onnx_path


def test_model_paths_exist() -> None:
    config = load_app_config()

    assert resolve_repo_path(config.resnet18.onnx_path).exists()
    assert resolve_repo_path(config.efficientad.onnx_path).exists()
    assert resolve_repo_path(config.efficientad.calibration_path).exists()


def test_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("SURFACE_DEFECT_MAX_UPLOAD_BYTES", "2048")
    monkeypatch.setenv("SURFACE_DEFECT_MIN_IMAGE_DIMENSION", "64")
    monkeypatch.setenv("SURFACE_DEFECT_MAX_IMAGE_DIMENSION", "1024")

    config = load_app_config()

    assert config.api_limits.max_upload_bytes == 2048
    assert config.api_limits.min_image_dimension == 64
    assert config.api_limits.max_image_dimension == 1024
