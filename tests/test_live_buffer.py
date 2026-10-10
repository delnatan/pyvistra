"""ImageWindow refresh behavior while an ImageBuffer is being written."""

import time

import numpy as np
from qtpy.QtWidgets import QApplication

from pyvistra import imshow
from pyvistra.data.buffer import ImageBuffer


def _wait_for(pred, timeout=2.0):
    app = QApplication.instance()
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.processEvents()
        if pred():
            return True
        time.sleep(0.005)
    return False


def test_projection_view_refreshes_on_write_to_other_z():
    """Regression: in Z-projection mode a write to any z other than the
    (hidden) current z_idx only repainted, so the projection stayed stale
    until the user moved a slider."""
    buf = ImageBuffer((1, 4, 1, 8, 8), np.uint16)
    win = imshow(buf, "live")
    try:
        win.view_state.set_z_projection(True)
        assert _wait_for(lambda: win.renderer.current_slice_cache is not None)
        assert win.z_idx != 3

        buf[0, 3, 0] = 50  # inside the projected range, not the current z

        assert _wait_for(lambda: win.renderer.current_slice_cache.max() == 50)
    finally:
        win.close()


def test_key_touches_current_slice_ranges():
    buf = ImageBuffer((2, 5, 1, 4, 4), np.uint16)
    win = imshow(buf, "keys")
    try:
        win.t_idx, win.z_idx = 1, 2
        touches = win._key_touches_current_slice
        assert touches((1, 2))
        assert touches((-1, slice(0, 5, 2)))  # negative t; stepped z hits 2
        assert not touches((1, slice(3, 5)))
        assert not touches((0, 2))

        win.view_state.set_z_projection(True)
        win.view_state.set_z_range(3, 4)
        assert touches((1, 4))
        assert not touches((1, 2))  # current z_idx, but outside the slab
    finally:
        win.close()
