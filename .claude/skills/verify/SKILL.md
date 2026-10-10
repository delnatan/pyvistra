---
name: verify
description: Manually run pyvistra's Qt/vispy GUI to observe a change end-to-end, beyond what the pytest suite covers.
---

# Verifying pyvistra GUI changes

pyvistra is a PyQt6 (via `qtpy`) + vispy desktop app. `tests/` includes
GUI-level regression tests (real windows via `imshow` under offscreen Qt,
e.g. `tests/test_live_buffer.py`, `tests/test_label_painting.py`), but
verifying a UI/viewer change still means constructing and driving the real
widgets through the paths a user would take, in a throwaway script.

## Recipe

```python
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # no display in CI/agent sessions

from qtpy.QtWidgets import QApplication
app = QApplication.instance() or QApplication(sys.argv)

# construct real widgets/viewers, e.g.:
from pyvistra.viewers.tiled import TiledViewer
viewer = TiledViewer(image_paths)
viewer.show()

# pump the event loop after anything async (background threads,
# queued signals, deferred deletes):
app.processEvents()
```

Run with the project venv: `.venv/bin/python <script>`.

- `QT_QPA_PLATFORM=offscreen` avoids needing a real display; vispy/GL
  logs a stream of harmless `QOpenGLWidget`/`createPlatformOpenGLContext`
  warnings under offscreen — ignore them, they don't affect assertions.
- For real mouse/keyboard interaction (not just calling methods
  directly), use `qtpy.QtTest.QTest.mouseClick(widget, Qt.LeftButton,
  Qt.NoModifier, QPoint(...))` / `QTest.keyClick(...)` so the actual
  event path (`mousePressEvent`, signal chains) executes, not just the
  handler in isolation.
- Background-thread work (e.g. `data/slice_loader.py`,
  `data/thumbnail_cache.py`) delivers via queued Qt signals — poll with
  a short `time.sleep` + `app.processEvents()` loop and a deadline
  rather than a single `processEvents()` call.
- Don't emit vispy canvas events (`canvas.events.mouse_press(...)`)
  under offscreen Qt: vispy's picking pass renders with GL and segfaults
  without a real context. Call the window's own handlers instead, e.g.
  `win.on_mouse_press(vispy.app.canvas.MouseEvent("mouse_press",
  pos=..., button=1))`, mapping image coords to canvas coords via
  `win.canvas.scene.node_transform(win.renderer.layers[0]).imap(...)`.
- Closed windows are deleted via `deleteLater`, which `processEvents()`
  alone doesn't run outside an event loop; also call
  `app.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)` when
  checking that something was freed.
- Real sample data: a CZI in `tests/data/`; anything else, generate.
- Generate throwaway test images with Pillow/numpy directly in the
  script (`tempfile.mkdtemp()`), don't rely on fixture files.
- Clean up temp dirs (`shutil.rmtree`) and call `viewer.close()` when
  done — some viewers own background threads (`ThumbnailDecodeWorker`,
  `SliceLoader`) that should be stopped via `closeEvent`.

## Worked example

`TiledViewer` fast-mode thumbnail grid (`pyvistra/viewers/tiled.py`,
`pyvistra/widgets/thumbnail_grid.py`, `pyvistra/data/thumbnail_cache.py`)
was verified this way: generated a folder of small PNGs, constructed
`TiledViewer(paths)`, asserted `viewer._fast_mode` and the concrete
widget type, waited for the background decode worker via
processEvents-in-a-loop, drove a real `QTest.mouseClick` to select a
tile, and paged/sorted/annotated through the real methods, checking
timing on a page revisit to confirm the decode cache actually avoided
re-decoding.
