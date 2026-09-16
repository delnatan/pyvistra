"""Tests for ConvergencePlotWidget's y-axis scaling in widgets/convergence_plot.py."""

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from qtpy.QtWidgets import QApplication

from pyvistra.widgets.convergence_plot import ConvergencePlotWidget

# Held at module level: an unreferenced QApplication is garbage-collected
# immediately, and constructing a widget without one aborts the process.
_app = QApplication.instance() or QApplication(sys.argv)


@pytest.fixture(scope="module")
def plot():
    # No teardown event processing: flushing the event loop here would run
    # other tests' pending deleteLater() calls (e.g. the shared
    # WindowManager) and break tests collected after this module.
    widget = ConvergencePlotWidget()
    widget.resize(400, 300)
    return widget


def _pixels(widget, label, axis_range):
    y_min, y_max, log_y_min, log_y_max = axis_range
    return [
        widget._y_to_px(v, y_min, y_max, 0, 100, log_y_min, log_y_max)
        for v in widget.series_data(label)
    ]


def test_negative_values_are_plotted_on_a_linear_axis(plot):
    """Regression test: in the default log-y mode, values <= 0 (e.g. a
    negative log-likelihood loss) used to be silently dropped."""
    plot.clear()
    plot.set_series("loss", [3.0, 0.5, 0.0, -1.2, -2.5])

    axis_range = plot._axis_range(list(plot._series.items()))
    y_min, y_max, log_y_min, _ = axis_range
    assert log_y_min is None
    assert y_min < -2.5 and y_max > 3.0

    pixels = _pixels(plot, "loss", axis_range)
    assert all(p is not None for p in pixels)
    assert pixels == sorted(pixels)  # decreasing values -> increasing pixel y


def test_positive_values_keep_log_axis(plot):
    plot.clear()
    plot.set_series("mse", [1.0, 1e-1, 1e-3])

    axis_range = plot._axis_range(list(plot._series.items()))
    assert axis_range[2] is not None
    assert all(p is not None for p in _pixels(plot, "mse", axis_range))


def test_axes_choose_their_scale_independently(plot):
    plot.clear()
    plot.set_series("loss", [1.0, -1.0], axis="left")
    plot.set_series("mse", [1.0, 1e-2], axis="right")

    items = list(plot._series.items())
    left = plot._axis_range([i for i in items if i[1]["axis"] == "left"])
    right = plot._axis_range([i for i in items if i[1]["axis"] == "right"])
    assert left[2] is None
    assert right[2] is not None


def test_non_finite_values_are_skipped_without_forcing_linear(plot):
    plot.clear()
    plot.set_series("mse", [1.0, float("nan"), float("inf"), 0.1])

    axis_range = plot._axis_range(list(plot._series.items()))
    assert axis_range[2] is not None
    pixels = _pixels(plot, "mse", axis_range)
    assert pixels[1] is None and pixels[2] is None
    assert pixels[0] is not None and pixels[3] is not None


def test_paints_mixed_sign_series_without_error(plot):
    plot.clear()
    plot.set_series("loss", [0.4, -0.3, -1.1], axis="left")
    plot.set_series("metric", [2.3, 2.2, 2.25], axis="right")
    assert not plot.grab().isNull()
