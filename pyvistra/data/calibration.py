"""Physical (scale-calibrated) <-> pixel-index coordinate conversion.

A window's ``meta["scale"]`` is ``(sz, sy, sx)`` physical pixel spacing (e.g.
microns/px). This module is the one place that turns a window's scale into
a coordinate conversion, so a shape's geometry can be measured on one window
and sampled on another with a different pixel size — converting the
*geometry* (a path, a center+radius) through a shared physical coordinate,
never resampling an already-rasterized/already-sampled result, which would
alias or blur it. ``pyvistra.widgets.line_profile``/``radial_profile_dialog``
established this pattern independently for their own shapes; this module is
the shared foundation other shape-driven tools (kymograph, region
statistics) reuse instead of reimplementing it a third/fourth time.

Qt-free, numpy/scipy only.
"""

from __future__ import annotations

import numpy as np


# vispy's Image visual draws pixel ``i`` over ``[i, i + 1)``, so in shape /
# world pixel coordinates (what the mouse and every ShapeRecord use) pixel
# centers sit at ``i + 0.5``. Array-index space (``map_coordinates``,
# ``np.mgrid``) puts them at ``i``. Every sampler or rasterizer that takes
# shape geometry must cross between the two through this offset, or results
# shift by half a pixel relative to what is drawn on screen.
PIXEL_CENTER_OFFSET = 0.5


def world_to_index(points_xy: np.ndarray) -> np.ndarray:
    """``(N, 2)`` world xy -> array-index xy (pixel centers at integers)."""
    return np.asarray(points_xy, dtype=float) - PIXEL_CENTER_OFFSET


def samples_for_length(length_px: float) -> int:
    """Default sample count for a path ``length_px`` pixels long: one per
    pixel step *including both endpoints* (``length + 1``), so a path drawn
    through pixel centers samples exactly those centers instead of
    stretching ``length`` samples across ``length`` pixels."""
    return max(2, int(np.ceil(length_px)) + 1)


def window_scale_yx(window) -> tuple[float, float] | None:
    """``(sy, sx)`` from ``window.meta["scale"]``, or ``None`` if the
    window (or the scale) isn't set — tolerates ``window=None`` so callers
    don't need a separate check for a not-yet-set source window.
    """
    scale = getattr(window, "meta", {}).get("scale") if window is not None else None
    if scale is None or len(scale) < 3:
        return None
    return float(scale[1]), float(scale[2])


def window_is_frequency_space(window) -> bool:
    """Whether ``window`` (or ``None``) is FFT/frequency-space output.

    See ``ImageWindow.pixel_space`` / ``fft_dialog.py``. A real-space
    distance/area and a frequency-space one aren't the same physical
    quantity, so callers comparing across windows should refuse rather than
    silently compare mismatched kinds of "space".
    """
    return getattr(window, "pixel_space", "real") == "frequency"


def _xy_scale(scale_yx: tuple[float, float]) -> np.ndarray:
    sy, sx = scale_yx
    return np.array([sx, sy], dtype=float)


def points_px_to_phys(points_xy: np.ndarray, scale_yx: tuple[float, float]) -> np.ndarray:
    """``(N, 2)`` xy pixel -> physical coordinates, each axis scaled
    independently (so anisotropic pixel sizes and off-axis paths are exact,
    not approximated by an averaged factor)."""
    return np.asarray(points_xy, dtype=float) * _xy_scale(scale_yx)


def points_phys_to_px(points_xy_phys: np.ndarray, scale_yx: tuple[float, float]) -> np.ndarray:
    """Inverse of :func:`points_px_to_phys` — converts a shared physical
    path into one specific window's own pixel grid."""
    return np.asarray(points_xy_phys, dtype=float) / _xy_scale(scale_yx)


def radius_px_to_phys(radius_px: float, scale_yx: tuple[float, float]) -> float:
    """Pixel radius -> physical radius.

    A circle has no natural anisotropic generalization (that would make it
    an ellipse), so this uses the average of the two axis scales — the same
    approximation already established by the radial-profile dialog.
    """
    sy, sx = scale_yx
    return float(radius_px) * 0.5 * (float(sy) + float(sx))


def radius_phys_to_px(radius_phys: float, scale_yx: tuple[float, float]) -> float:
    """Inverse of :func:`radius_px_to_phys`."""
    sy, sx = scale_yx
    return float(radius_phys) / (0.5 * (float(sy) + float(sx)))


def sample_along_path(
    image_2d: np.ndarray, path_xy: np.ndarray, *, order: int = 1, mode: str = "nearest"
) -> np.ndarray:
    """Sample ``image_2d`` at ``(N, 2)`` world xy pixel coordinates (the
    shape/mouse frame, see :data:`PIXEL_CENTER_OFFSET`) via
    ``scipy.ndimage.map_coordinates`` (which takes ``[rows, cols]`` =
    ``[y, x]`` — this is the one place that transpose happens)."""
    from scipy.ndimage import map_coordinates

    path_xy = world_to_index(path_xy)
    xs = path_xy[:, 0]
    ys = path_xy[:, 1]
    coords = np.array([ys, xs], dtype=float)
    return map_coordinates(np.asarray(image_2d, dtype=float), coords, order=order, mode=mode)
