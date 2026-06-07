from __future__ import annotations

import pytest

from defect_detection.monitoring.drift import compute_population_stability_index


def test_population_stability_index_is_small_for_similar_distributions() -> None:
    reference = [0.1, 0.2, 0.3, 0.4, 0.5] * 20
    candidate = [0.1, 0.22, 0.31, 0.39, 0.5] * 20

    psi = compute_population_stability_index(reference, candidate)
    assert psi >= 0.0
    assert psi < 0.2


def test_population_stability_index_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        compute_population_stability_index([], [0.1, 0.2])
