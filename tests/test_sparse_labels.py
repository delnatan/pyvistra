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


def test_to_dense_does_not_wrap_large_label_ids():
    """Regression: to_dense() defaulted to uint16, so label 65537 came out
    as 1 -- silently merged with a different object."""
    labels = _labels_with((1, ([0], [0])), (65537, ([1], [1])))

    dense = labels.to_dense()

    assert dense.dtype == np.uint32
    assert dense[0, 0] == 1 and dense[1, 1] == 65537
    assert _labels_with((7, ([0], [0]))).to_dense().dtype == np.uint16  # unchanged default
    with pytest.raises(ValueError, match="does not fit"):
        labels.to_dense(np.uint16)


@pytest.mark.parametrize("ys, xs", [([-1], [3]), ([3], [10])])
def test_add_pixels_rejects_out_of_bounds(ys, xs):
    """Regression: add()/update() validated bounds but add_pixels() didn't;
    a -1 then wrapped onto the opposite edge in to_dense()."""
    labels = _labels_with((1, ([0], [0])))

    with pytest.raises(ValueError, match="out of bounds"):
        labels.add_pixels(1, (np.array(ys), np.array(xs)))
    with pytest.raises(ValueError, match="out of bounds"):
        labels.add_pixels(2, (np.array(ys), np.array(xs)))  # new label too

    assert labels.labels == [1] and labels.area(1) == 1


def test_npz_roundtrip_keeps_metadata_like_zarr(tmp_path):
    """Regression: .sparse.npz dropped the per-label metadata that
    .sparse.zarr preserves (so zarr -> npz conversion lost it)."""
    labels = _labels_with((1, ([0], [0])), (4, ([2, 3], [2, 3])))
    labels._metadata = {4: {"class": "nucleus", "score": 0.9}}

    for name in ("x.sparse.zarr", "x.sparse.npz"):
        labels.save(tmp_path / name)
        loaded = SparseLabels.from_file(tmp_path / name)
        assert loaded._metadata == {4: {"class": "nucleus", "score": 0.9}}
        assert loaded.shape == (10, 10) and type(loaded.shape[0]) is int
        np.testing.assert_array_equal(loaded.to_dense(), labels.to_dense())


# ---- change notification ----


def _recording(labels):
    events = []
    labels.subscribe(lambda kind, label: events.append((kind, label)))
    return events


def test_every_mutation_notifies_after_it_lands():
    labels = SparseLabels((10, 10))
    seen_area = []
    labels.subscribe(lambda kind, label: seen_area.append(labels.area(label) if label in labels else 0))
    events = _recording(labels)
    one = (np.array([1, 1]), np.array([1, 2]))
    labels.add(1, one)
    labels.add_pixels(1, (np.array([3]), np.array([3])))
    labels.add_pixels(2, (np.array([5]), np.array([5])))  # creates label 2
    labels.update(2, (np.array([6]), np.array([6])))
    labels.remove_pixels(1, (np.array([3]), np.array([3])))
    labels.relabel(2, 7)
    labels.merge([1, 7])
    labels.add(9, (np.array([8]), np.array([8])))
    labels.remove(9)
    labels.clear()
    assert events == [
        ("added", 1), ("painted", 1), ("painted", 2), ("updated", 2), ("erased", 1),
        ("relabeled", 7), ("merged", 1), ("added", 9), ("removed", 9), ("cleared", -1),
    ]
    assert seen_area[:2] == [2, 3]  # the change is visible to the subscriber


def test_failed_mutation_does_not_notify():
    labels = SparseLabels((10, 10))
    events = _recording(labels)
    with pytest.raises(KeyError):
        labels.remove(3)
    with pytest.raises(ValueError):
        labels.add(0, (np.array([1]), np.array([1])))
    assert events == []


def test_unsubscribe_and_a_failing_subscriber():
    labels = SparseLabels((10, 10))

    def broken(kind, label):
        raise RuntimeError("boom")

    labels.subscribe(broken)
    events = []
    unsubscribe = labels.subscribe(lambda kind, label: events.append(kind))
    labels.add(1, (np.array([1]), np.array([1])))
    unsubscribe()
    labels.remove(1)
    assert events == ["added"]


def test_copies_and_loaded_labels_start_without_subscribers(tmp_path):
    labels = SparseLabels((10, 10))
    labels.add(1, (np.array([1]), np.array([1])))
    events = _recording(labels)
    copy = labels.copy()
    copy.add(2, (np.array([2]), np.array([2])))
    labels.save(tmp_path / "l.sparse.npz")
    loaded = SparseLabels.from_file(tmp_path / "l.sparse.npz")
    loaded.add(3, (np.array([3]), np.array([3])))
    assert events == []
