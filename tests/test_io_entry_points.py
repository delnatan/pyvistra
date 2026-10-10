"""Smoke tests for small public I/O entry points that had no coverage."""

import numpy as np
import pytest
import tifffile

import pyvistra
from pyvistra.data.buffer import ImageBuffer
from pyvistra.data.labels import SparseLabels
from pyvistra.io import load_image, save_tiff


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



@pytest.mark.parametrize(
    "shape",
    [(3, 1, 1, 8, 8), (1, 1, 3, 8, 8), (3, 1, 2, 8, 8), (2, 3, 1, 8, 8), (2, 3, 4, 8, 8)],
)
def test_tiff_roundtrip_preserves_axes(tmp_path, shape):
    """Regression: the TIFF loader ignored the file's axis labels and
    guessed from the squeezed shape, so a timelapse (TYX) or channel stack
    (CYX) reloaded as a Z-stack and TZYX reloaded as ZCYX."""
    data = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    path = tmp_path / "x.tif"
    save_tiff(str(path), data)

    img, _meta = load_image(str(path))

    assert img.shape == shape
    assert np.array_equal(np.asarray(img[:, :, :, :, :]), data)


def test_tiff_without_axis_labels_keeps_heuristic(tmp_path):
    path = tmp_path / "plain.tif"
    tifffile.imwrite(path, np.zeros((5, 8, 8), np.uint16))  # axes "QYX"

    img, _meta = load_image(str(path))

    assert img.shape == (1, 5, 1, 8, 8)  # unlabelled 3D -> Z-stack, as before


def test_tiff_explicit_dims_override_file_axes(tmp_path):
    path = tmp_path / "t.tif"
    save_tiff(str(path), np.zeros((3, 1, 1, 8, 8), np.uint16))  # file says TYX

    img, _meta = load_image(str(path), dims="zyx")

    assert img.shape == (1, 3, 1, 8, 8)
