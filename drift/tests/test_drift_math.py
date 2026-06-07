from drift.detect_drift import relative_change


def test_relative_change_positive() -> None:
    assert relative_change(100.0, 120.0) == 0.2


def test_relative_change_negative() -> None:
    assert relative_change(100.0, 80.0) == -0.2


def test_relative_change_zero_baseline() -> None:
    assert relative_change(0.0, 10.0) == 0.0
