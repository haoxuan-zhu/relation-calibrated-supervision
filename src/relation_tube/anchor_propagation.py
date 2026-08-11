"""Label-preserving helpers for relation propagation."""

from __future__ import annotations

import numpy as np


def preserve_observed_targets(
    propagated: np.ndarray,
    observed_indices: np.ndarray,
    observed_targets: np.ndarray,
) -> np.ndarray:
    """Keep measured targets on labeled rows and propagate only to unlabeled rows."""

    propagated = np.asarray(propagated, dtype=np.float64).reshape(-1)
    observed_indices = np.asarray(observed_indices, dtype=np.int64).reshape(-1)
    observed_targets = np.asarray(observed_targets, dtype=np.float64).reshape(-1)
    if len(observed_indices) != len(observed_targets):
        raise ValueError("observed indices and targets have different lengths")
    if len(np.unique(observed_indices)) != len(observed_indices):
        raise ValueError("observed indices contain duplicates")
    if np.any(observed_indices < 0) or np.any(observed_indices >= len(propagated)):
        raise IndexError("observed index is outside the propagated target array")
    result = propagated.copy()
    result[observed_indices] = observed_targets
    return result
