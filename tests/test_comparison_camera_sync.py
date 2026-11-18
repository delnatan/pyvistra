"""Linked pan/zoom in a ComparisonPair follows the camera, not mouse events.

Regression: the pair (and each window's overlay) used to listen on
canvas mouse_wheel/mouse_release, which vispy delivers *before* the camera
moves -- so the partner always lagged one step and overlays drew for the
previous view.
"""

import numpy as np
import pytest

from pyvistra import imshow
from pyvistra.ui.comparison import ComparisonPair


@pytest.fixture
def pair():
    a = imshow(np.random.rand(1, 1, 1, 64, 64).astype(np.float32), "a")
    b = imshow(np.random.rand(1, 1, 1, 64, 64).astype(np.float32), "b")
    p = ComparisonPair(a, b)
    yield p
    p.end()
    a.close()
    b.close()


def test_programmatic_zoom_propagates_and_updates_overlay(pair):
    a, b = pair.window_a, pair.window_b
    pair.set_link_pan_zoom(True)
    b.overlay.set_config({"show_scale_bar": True, "scale_bar_unit": "px",
                          "scale_bar_length": 10})
    before = np.ptp(b.overlay._line.pos[:, 0])

    a.view.camera.zoom(0.5)

    assert tuple(b.view.camera.rect.pos) == pytest.approx(tuple(a.view.camera.rect.pos))
    assert b.view.camera.rect.size == pytest.approx(a.view.camera.rect.size)
    # Zooming in 2x doubles the on-screen scale bar.
    assert np.ptp(b.overlay._line.pos[:, 0]) == pytest.approx(2 * before)
