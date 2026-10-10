"""Data-ownership contract between windows and the proxies they display.

``ImageWindow`` takes over one reference to a proxy it is given, so a
caller who keeps using the proxy must acquire on the window's behalf.
``imshow`` is the public entry point where that's always the case.
"""

import gc
import weakref

import numpy as np
from qtpy.QtCore import QEvent
from qtpy.QtWidgets import QApplication

from pyvistra import imshow
from pyvistra.data.buffer import ImageBuffer
from pyvistra.data.proxies import Numpy5DProxy


def _flush_deletes():
    app = QApplication.instance()
    for _ in range(3):
        app.processEvents()
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
        gc.collect()


def test_closing_imshow_window_keeps_callers_buffer():
    """Regression: closing the window used to delete the caller's buffer
    store, after which reads silently returned zeros (zarr fill value)."""
    buf = ImageBuffer((1, 1, 1, 8, 8), np.uint16)
    buf[0, 0, 0] = 5

    win = imshow(buf, "buffer")
    win.close()
    del win
    _flush_deletes()

    assert buf._path.exists()
    assert np.all(buf[0, 0, 0] == 5)

    # The caller's own reference still cleans up normally.
    path = buf._path
    del buf
    gc.collect()
    assert not path.exists()


def test_toolbar_forgets_closed_windows():
    """Regression: Toolbar.open_windows kept every window it ever opened,
    pinning each closed window's image data in memory for the session."""
    from pyvistra.ui.toolbar import Toolbar

    toolbar = Toolbar()
    data = Numpy5DProxy(np.zeros((1, 1, 1, 16, 16), np.uint16))
    data_ref = weakref.ref(data)

    worker = object()
    toolbar._pending_loads[worker] = None
    toolbar._on_image_loaded(data, {"filename": "x"}, "x.tif", worker)
    del data
    assert len(toolbar.open_windows) == 1

    toolbar.open_windows[0].close()
    _flush_deletes()

    assert toolbar.open_windows == []
    assert data_ref() is None
    toolbar.close()


def test_window_title_shows_small_pixel_sizes(tmp_path):
    """Regression: the title used .2f, so a 0.065 um pixel read "0.06"."""
    import tifffile

    path = tmp_path / "cal.tif"
    tifffile.imwrite(
        path, np.zeros((8, 8), np.uint16), imagej=True,
        resolution=(1 / 0.065, 1 / 0.065), metadata={"unit": "um"},
    )
    from pyvistra.ui.window import ImageWindow

    win = ImageWindow(str(path))
    try:
        assert "[0.065 x 0.065 µm]" in win.windowTitle()
    finally:
        win.close()
