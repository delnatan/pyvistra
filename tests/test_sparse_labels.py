"""SparseLabels data-model invariants (data/labels.py)."""

import numpy as np
import pytest

from pyvistra.data.labels import SparseLabels


def _labels_with(*objects, shape=(10, 10)):
    labels = SparseLabels(shape)
    for label, (ys, xs) in objects:
        labels.add(label, (np.array(ys), np.array(xs)))
    return labels


def test_merge_refuses_to_overwrite_unrelated_label():
    """Regression: merge([1, 2], new_label=3) replaced existing object 3's
    pixels with the merged ones, silently discarding object 3."""
    labels = _labels_with((1, ([0], [0])), (2, ([1], [1])), (3, ([5], [5])))

    with pytest.raises(ValueError, match="already exists"):
        labels.merge([1, 2], new_label=3)

    assert labels.labels == [1, 2, 3]
    assert labels.area(3) == 1


def test_merge_into_member_label_and_fresh_label():
    labels = _labels_with((1, ([0], [0])), (2, ([1], [1])))
    assert labels.merge([1, 2], new_label=2) == 2
    assert labels.labels == [2] and labels.area(2) == 2

    labels = _labels_with((1, ([0], [0])), (2, ([1], [1])))
    assert labels.merge([1, 2], new_label=7) == 7
    assert labels.labels == [7] and labels.area(7) == 2
