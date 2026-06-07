"""
Simple score-distribution drift utilities.
"""

from __future__ import annotations

import numpy as np


def compute_population_stability_index(
    reference: list[float] | np.ndarray,
    candidate: list[float] | np.ndarray,
    *,
    bins: int = 10,
) -> float:
    """Compute PSI between two score distributions."""
    reference_array = np.asarray(reference, dtype=float)
    candidate_array = np.asarray(candidate, dtype=float)

    if reference_array.size == 0 or candidate_array.size == 0:
        raise ValueError("Reference and candidate arrays must be non-empty.")

    combined_min = float(min(reference_array.min(), candidate_array.min()))
    combined_max = float(max(reference_array.max(), candidate_array.max()))

    if np.isclose(combined_min, combined_max):
        return 0.0

    effective_bins = min(
        bins,
        max(
            2,
            min(
                np.unique(reference_array).size,
                np.unique(candidate_array).size,
            ),
        ),
    )

    bin_edges = np.linspace(combined_min, combined_max, effective_bins + 1)

    reference_hist, _ = np.histogram(reference_array, bins=bin_edges)
    candidate_hist, _ = np.histogram(candidate_array, bins=bin_edges)

    reference_ratio = np.clip(reference_hist / reference_array.size, 1e-6, None)
    candidate_ratio = np.clip(candidate_hist / candidate_array.size, 1e-6, None)

    psi = np.sum((candidate_ratio - reference_ratio) * np.log(candidate_ratio / reference_ratio))
    return float(psi)
