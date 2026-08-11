import numpy as np
import pytest

from relation_tube.anchor_propagation import preserve_observed_targets


def test_preserve_observed_targets_changes_only_anchor_rows() -> None:
    propagated = np.asarray([0.1, 0.2, 0.3, 0.4])
    result = preserve_observed_targets(
        propagated, np.asarray([1, 3]), np.asarray([-1.0, 2.0])
    )
    np.testing.assert_allclose(result, [0.1, -1.0, 0.3, 2.0])
    np.testing.assert_allclose(propagated, [0.1, 0.2, 0.3, 0.4])


def test_preserve_observed_targets_rejects_duplicate_or_invalid_rows() -> None:
    propagated = np.zeros(3)
    with pytest.raises(ValueError, match="duplicates"):
        preserve_observed_targets(propagated, np.asarray([1, 1]), np.asarray([0.0, 1.0]))
    with pytest.raises(IndexError, match="outside"):
        preserve_observed_targets(propagated, np.asarray([3]), np.asarray([0.0]))
