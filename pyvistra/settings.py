"""User-editable settings for pyvistra, stored at ``~/.pyvistra/settings.json``.

Follows the same ``~/.pyvistra/`` convention as the buffer scratch directory
in :mod:`pyvistra.data.buffer`. The file is created with defaults on first
read if it doesn't exist yet, so there's always something for the user to
find and edit.
"""

import json
from pathlib import Path

SETTINGS_DIR = Path.home() / ".pyvistra"
SETTINGS_FILE = SETTINGS_DIR / "settings.json"

DEFAULTS = {
    # Files estimated to fit under this size load fully into memory
    # automatically; at or above it, the user is asked. See
    # pyvistra.io.estimate_load_bytes and pyvistra.ui.toolbar.spawn_viewer.
    "in_memory_threshold_mb": 200,
}


def load_settings():
    """Return the current settings, creating the file with defaults if
    it doesn't exist yet. Unknown/missing keys fall back to DEFAULTS."""
    if not SETTINGS_FILE.exists():
        SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps(DEFAULTS, indent=2) + "\n")
        return dict(DEFAULTS)

    try:
        data = json.loads(SETTINGS_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return dict(DEFAULTS)

    return {**DEFAULTS, **data}


def get_in_memory_threshold_bytes():
    """Files below this size (in bytes) load into memory automatically."""
    return int(load_settings()["in_memory_threshold_mb"] * 1024 * 1024)
