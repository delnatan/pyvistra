"""
ImageBuffer — zarr-backed 5D write buffer for streaming image operations.

The intended workflow is::

    buf = ImageBuffer(shape=(T, Z, C, Y, X), dtype=np.uint16, metadata=meta)

    # In a worker thread / loop:
    for t in range(T):
        result = heavy_computation(source[t, ...])
        buf[t, ...] = result

    # Meanwhile the UI can display buf live:
    win = imshow(buf, title="Processing…")

Live updates are wired automatically: ``ImageWindow`` subscribes to the
buffer via ``subscribe(callback)`` and refreshes the view whenever a write
touches the currently displayed (t, z) slice.  Callbacks fire on the
writer's thread; consumers that need GUI-thread execution should marshal
via a queued Qt signal (the ``ImageWindow`` integration does this).

After processing, ``buf`` can be saved or kept in memory.  The underlying
zarr store is a temporary directory in ``~/.pyvistra/buffers/<pid>/`` that
is deleted when ``buf.close()`` is called (or when the object is garbage-
collected and the refcount reaches zero). Python doesn't guarantee that
happens at exit, so each process also removes its own ``<pid>/`` directory
on exit, and :func:`sweep_stale_buffers` reclaims directories left behind
by processes that crashed.
"""

import atexit
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import zarr

from .proxies import RefCountMixin

BUFFER_DIR = Path.home() / ".pyvistra" / "buffers"

# Pre-<pid>/ layout: stores sat directly in BUFFER_DIR with no owner info.
# Only an old pyvistra version can still be writing one, so they're swept
# once they're clearly stale rather than immediately.
_LEGACY_MAX_AGE_S = 24 * 3600

_session_dir = None


def _pid_alive(pid):
    """Whether process ``pid`` is running. Errs toward True (keep)."""
    if pid == os.getpid():
        return True
    if sys.platform == "win32":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:  # e.g. PermissionError: exists, owned by someone else
        return True
    return True


def sweep_stale_buffers(buffer_dir=None):
    """Delete scratch buffers whose owning process is gone.

    Safe to call while other pyvistra processes run: their ``<pid>/``
    directories are kept. Legacy top-level ``*.zarr`` stores are removed
    once older than a day. Returns the number of entries removed.
    """
    root = Path(buffer_dir) if buffer_dir is not None else BUFFER_DIR
    if not root.is_dir():
        return 0
    removed = 0
    now = time.time()
    for entry in root.iterdir():
        try:
            if entry.name.isdigit():
                stale = not _pid_alive(int(entry.name))
            elif entry.suffix == ".zarr":
                stale = now - entry.stat().st_mtime > _LEGACY_MAX_AGE_S
            else:
                continue
            if stale:
                shutil.rmtree(entry, ignore_errors=True)
                removed += 1
        except OSError:
            continue
    return removed


def _remove_session_dir(path):
    shutil.rmtree(path, ignore_errors=True)


def _buffer_session_dir():
    """This process's ``BUFFER_DIR/<pid>/``. On first use, sweeps stale
    buffers from earlier crashed sessions and registers removal of this
    process's own directory at exit."""
    global _session_dir
    if _session_dir is None:
        sweep_stale_buffers()
        _session_dir = BUFFER_DIR / str(os.getpid())
        _session_dir.mkdir(parents=True, exist_ok=True)
        atexit.register(_remove_session_dir, _session_dir)
    return _session_dir


class ImageBuffer(RefCountMixin):
    """
    Zarr-backed 5D array with the same read interface as the proxies plus
    write support via ``__setitem__`` and change notifications via
    ``subscribe()``.

    Shape must be 5D ``(T, Z, C, Y, X)``.
    """

    def __init__(self, shape, dtype, chunks=None, metadata=None):
        self._path = _buffer_session_dir() / f"{uuid.uuid4()}.zarr"

        T, Z, C, Y, X = shape
        if chunks is None:
            chunks = (1, min(16, Z), C, min(512, Y), min(512, X))

        self._store = zarr.open(
            str(self._path),
            mode="w",
            shape=shape,
            dtype=dtype,
            chunks=chunks,
        )

        self.metadata = metadata or {}
        self.ndim = 5
        self._listeners = []
        self._init_refcount()

    @property
    def shape(self):
        return self._store.shape

    @property
    def dtype(self):
        return self._store.dtype

    def __getitem__(self, key):
        return np.asarray(self._store[key])

    def __setitem__(self, key, value):
        self._store[key] = value
        if self._listeners:
            for cb in list(self._listeners):
                try:
                    cb(key)
                except Exception:
                    pass

    def subscribe(self, callback):
        """Register *callback(key)* fired on every ``__setitem__``.

        *key* is the slicing tuple as passed to ``__setitem__`` (not
        normalised). Callbacks run on the writer's thread.

        Returns a zero-arg unsubscribe function.
        """
        self._listeners.append(callback)

        def _unsubscribe():
            try:
                self._listeners.remove(callback)
            except ValueError:
                pass

        return _unsubscribe

    def save_as(self, filepath):
        """Export buffer contents to an ImageJ-hyperstack TIFF."""
        from ..io import save_tiff
        scale = self.metadata.get("scale", (1.0, 1.0, 1.0))
        save_tiff(filepath, self._store[:], scale=scale, metadata=self.metadata)

    def close(self):
        if self._path.exists():
            shutil.rmtree(self._path)

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass
