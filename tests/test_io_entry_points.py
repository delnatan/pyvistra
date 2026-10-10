"""Smoke tests for small public I/O entry points that had no coverage."""

import numpy as np
import tifffile

import pyvistra
from pyvistra.data.buffer import ImageBuffer
from pyvistra.data.labels import SparseLabels


def test_sparse_labels_roundtrip_via_package_functions(tmp_path):
    """Regression: pyvistra.load_sparse_labels imported a non-existent
    ``pyvistra.labels`` module."""
    labels = SparseLabels((10, 10))
    labels.add(3, (np.array([1, 2]), np.array([4, 5])))
    path = tmp_path / "x.sparse.npz"

    pyvistra.save_sparse_labels(path, labels)
    loaded = pyvistra.load_sparse_labels(path)

    assert loaded.labels == [3]
    assert np.array_equal(loaded.to_dense(), labels.to_dense())


def test_image_buffer_save_as_writes_tiff(tmp_path):
    """Regression: ImageBuffer.save_as imported ``pyvistra.data.io``."""
    buf = ImageBuffer((2, 1, 1, 4, 4), np.uint16, metadata={"scale": (1, 0.5, 0.5)})
    buf[1, 0, 0] = 9
    path = tmp_path / "buf.tif"

    buf.save_as(str(path))

    assert tifffile.imread(path)[1].max() == 9
