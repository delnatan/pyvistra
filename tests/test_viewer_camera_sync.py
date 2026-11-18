"""Ortho and tiled viewers sync cameras from the camera's own transform.

Regression: both listened on canvas mouse_wheel/mouse_release, which vispy
delivers *before* the camera moves, so synced views lagged one step.
Programmatic camera changes (which emit no mouse event) now sync too.
"""

import types

import numpy as np
import pytest
from qtpy.QtWidgets import QApplication

from pyvistra.data.proxies import Numpy5DProxy
from pyvistra.viewers.ortho import OrthoViewer
from pyvistra.viewers.tiled import TileWidget


@pytest.fixture(autouse=True)
def _app():
    from pyvistra import imshow  # noqa: F401  (ensures QApplication setup)

    return QApplication.instance() or QApplication([])


def test_ortho_zoom_syncs_shared_axes():
    data = Numpy5DProxy(np.random.rand(1, 8, 1, 32, 48).astype(np.float32))
    viewer = OrthoViewer(data)
    try:
        viewer.view_yx.camera.zoom(0.5)
        yx = viewer.view_yx.camera.rect
        zy = viewer.view_zy.camera.rect
        zx = viewer.view_zx.camera.rect
        # ZY shares Y (vertical) with YX; ZX shares X (horizontal).
        assert (zy.bottom, zy.height) == pytest.approx((yx.bottom, yx.height))
        assert (zx.left, zx.width) == pytest.approx((yx.left, yx.width))
    finally:
        viewer.close()


def test_tile_reports_new_rect_but_not_its_own_fit():
    tile = TileWidget()
    seen = []
    tile.set_camera_callback(lambda t: seen.append(tuple(t.get_camera_rect().size)))

    tile.view.camera.zoom(0.5)
    assert seen == [pytest.approx(tuple(tile.view.camera.rect.size))]

    # Fitting is per-tile and must not be broadcast to other tiles.
    tile.data = types.SimpleNamespace(shape=(1, 1, 1, 20, 30))
    tile._fit_view()
    assert len(seen) == 1
    tile.close()
