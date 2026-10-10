"""Scratch-buffer lifecycle: per-process dirs, exit cleanup, stale sweep."""

import os
import subprocess
import sys
import time

from pyvistra.data.buffer import sweep_stale_buffers


def _dead_pid():
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_sweep_removes_only_stale_entries(tmp_path):
    dead = tmp_path / str(_dead_pid())
    live = tmp_path / str(os.getpid())
    old_legacy = tmp_path / "0a1b.zarr"
    new_legacy = tmp_path / "2c3d.zarr"
    other = tmp_path / "notes.txt"
    for d in (dead, live, old_legacy, new_legacy):
        (d / "x.zarr").mkdir(parents=True)
    other.write_text("keep")
    day_ago = time.time() - 2 * 24 * 3600
    os.utime(old_legacy, (day_ago, day_ago))

    assert sweep_stale_buffers(tmp_path) == 2

    assert not dead.exists() and not old_legacy.exists()
    assert live.exists() and new_legacy.exists() and other.exists()


def _run(script, home, **env):
    env = dict(os.environ, HOME=str(home), USERPROFILE=str(home), **env)
    subprocess.run([sys.executable, "-c", script], env=env, check=True)


def test_next_session_reclaims_buffers_from_crashed_session(tmp_path):
    """Regression: buffers were only deleted by __del__/close, so any
    session that died hard (crash, segfault on quit, os._exit) leaked its
    stores for good -- ~1 GB had accumulated in ~/.pyvistra/buffers."""
    crash = (
        "import os, numpy as np\n"
        "from pyvistra.data.buffer import ImageBuffer\n"
        "buf = ImageBuffer((1, 1, 1, 8, 8), np.uint16)\n"
        "os._exit(0)\n"
    )
    _run(crash, tmp_path)
    buffers = tmp_path / ".pyvistra" / "buffers"
    assert len(list(buffers.iterdir())) == 1  # the crashed session's dir

    _run(
        "import numpy as np\n"
        "from pyvistra.data.buffer import ImageBuffer\n"
        "ImageBuffer((1, 1, 1, 4, 4), np.uint8).close()\n",
        tmp_path,
    )
    assert list(buffers.iterdir()) == []
