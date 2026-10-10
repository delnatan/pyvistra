"""Zarr / OME-Zarr loading (io.load_zarr + data.proxies.Zarr5DProxy)."""

import itertools
import os

import numpy as np
import pytest
import zarr

from pyvistra.data.proxies import Zarr5DProxy
from pyvistra.io import load_zarr


def _ome_multiscale(axes, scale, units):
    return {
        "version": "0.4",
        "axes": [
            {"name": a, **({"unit": units[a]} if a in units else {})} for a in axes
        ],
        "datasets": [
            {"path": "0", "coordinateTransformations": [{"type": "scale", "scale": scale}]}
        ],
    }


def _write_ome(path, data_tczyx, *, version):
    axes = "tczyx"
    ms = _ome_multiscale(axes, [1.0, 1.0, 0.5, 200.0, 200.0], {"z": "micrometer", "y": "nanometer", "x": "nanometer"})
    omero = {"channels": [{"label": "DAPI"}, {"label": "GFP"}]}
    if version == "0.4":
        root = zarr.open_group(str(path), mode="w", zarr_format=2)
        root.attrs.update({"multiscales": [ms], "omero": omero})
    else:
        root = zarr.open_group(str(path), mode="w", zarr_format=3)
        root.attrs.update({"ome": {"version": "0.5", "multiscales": [ms], "omero": omero}})
    root.create_array("0", data=data_tczyx)


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_ome_zarr_axes_scale_and_channels(tmp_path, version):
    """Regression: OME-Zarr stores (t, c, z, y, x) but the proxy assumed
    pyvistra's (t, z, c, y, x), swapping channels and Z."""
    T, C, Z, Y, X = 2, 2, 3, 4, 5
    data = np.arange(T * C * Z * Y * X, dtype=np.uint16).reshape(T, C, Z, Y, X)
    path = tmp_path / "img.ome.zarr"
    _write_ome(path, data, version=version)

    proxy, meta = load_zarr(str(path))

    assert proxy.shape == (T, Z, C, Y, X)
    np.testing.assert_array_equal(proxy[1, 2, 0], data[1, 0, 2])
    np.testing.assert_array_equal(proxy[:, :, :, :, :], data.transpose(0, 2, 1, 3, 4))
    assert meta["scale"] == pytest.approx((0.5, 0.2, 0.2))  # nm -> um
    assert [c["name"] for c in meta["channels"]] == ["DAPI", "GFP"]


@pytest.mark.parametrize("axes", ["yx", "zyx", "czyx", "tyx", "tczyx", "xy"])
def test_zarr_proxy_matches_5d_numpy_for_any_axes(axes):
    """The proxy must slice exactly like the equivalent 5D ndarray, for
    every axis order and every int/slice mix -- including singleton axes
    the stored array doesn't have."""
    sizes = dict(t=2, z=3, c=2, y=4, x=5)
    shape = tuple(sizes[a] for a in axes)
    stored = np.arange(np.prod(shape)).reshape(shape)
    full = stored.transpose([axes.index(a) for a in "tzcyx" if a in axes])
    full = full.reshape(tuple(sizes[a] if a in axes else 1 for a in "tzcyx"))
    proxy = Zarr5DProxy(zarr.array(stored), axes)

    assert proxy.shape == full.shape
    picks = [0, slice(None), slice(0, 1)]
    for t, z, c in itertools.product(picks, repeat=3):
        key = (t, z, c, slice(None), slice(1, 3))
        np.testing.assert_array_equal(proxy[key], full[key])
    np.testing.assert_array_equal(proxy[..., 1], full[..., 1])


def test_plain_zarr_keeps_shape_heuristic(tmp_path):
    path = tmp_path / "plain.zarr"
    zarr.save_array(str(path), np.zeros((3, 4, 5), np.uint8))

    proxy, meta = load_zarr(str(path))

    assert proxy.shape == (1, 3, 1, 4, 5)  # 3D -> Z-stack, as before
    assert meta["filename"] == "plain.zarr"


def _make_zarr(path):
    zarr.save_array(str(path), np.full((2, 4, 5), 7, np.uint8))


def test_load_image_accepts_zarr_dir_without_trailing_slash(tmp_path):
    """Regression: only "x.zarr/" was recognized; "x.zarr" (QFileDialog,
    os.walk, scripts) fell through to the TIFF loader and failed."""
    from pyvistra.io import estimate_load_bytes, load_image

    path = tmp_path / "img.zarr"
    _make_zarr(path)

    for p in (str(path), str(path) + "/"):
        img, meta = load_image(p)
        assert img.shape == (1, 2, 1, 4, 5)
        assert meta["filename"] == "img.zarr"
        assert estimate_load_bytes(p) == 2 * 4 * 5


def test_collect_image_files_finds_nested_zarr(tmp_path):
    """Regression: os.walk directory names never end in "/", so zarr
    images inside a dropped folder were walked into instead of opened."""
    from pyvistra.data.labels import SparseLabels
    from pyvistra.ui.toolbar import collect_image_files

    folder = tmp_path / "exp"
    (folder / "sub").mkdir(parents=True)
    _make_zarr(folder / "a.zarr")
    _make_zarr(folder / "sub" / "b.zarr")
    SparseLabels((4, 5)).save(folder / "a.sparse.zarr")  # labels, not an image
    (folder / "c.tif").write_bytes(b"")

    found = collect_image_files([str(folder)])

    rel = sorted(os.path.relpath(p, folder) for p in found)
    assert rel == ["a.zarr", "c.tif", os.path.join("sub", "b.zarr")]
