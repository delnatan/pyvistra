"""Tests for readers/imaris_writer.py."""

import h5py
import numpy as np

from pyvistra.readers.imaris_writer import save_imaris


def _read_unit_attr(filepath):
    with h5py.File(filepath, "r") as f:
        raw = f["DataSetInfo/Image"].attrs["Unit"]
    return b"".join(raw).decode("utf-8")


def test_save_imaris_labels_frequency_space_scale_correctly(tmp_path):
    """FFT output tags itself space="frequency" (see fft_dialog.py) --
    saving it to file must not silently claim its cycles/um scale is a
    real-space "um" pixel size.
    """
    data = np.zeros((1, 1, 1, 8, 8), dtype=np.float32)
    out = tmp_path / "fft_result.ims"

    save_imaris(
        str(out),
        data,
        metadata={"scale": (1.0, 0.1, 0.1), "space": "frequency"},
        resolution_levels=False,
    )

    assert _read_unit_attr(str(out)) == "1/um"


def test_save_imaris_labels_real_space_scale_as_um(tmp_path):
    data = np.zeros((1, 1, 1, 8, 8), dtype=np.float32)
    out = tmp_path / "real_space.ims"

    save_imaris(
        str(out),
        data,
        metadata={"scale": (1.0, 0.1, 0.1)},
        resolution_levels=False,
    )

    assert _read_unit_attr(str(out)) == "um"


def _attr(f, path, key):
    return b"".join(f[path].attrs[key]).decode("utf-8")


def test_save_imaris_handles_nan_in_float_data(tmp_path):
    """Regression: a single NaN made the histogram range non-finite and the
    save raised ValueError."""
    from pyvistra.io import load_image

    data = np.random.rand(1, 2, 1, 8, 8).astype(np.float32)
    data[0, 0, 0, 0, 0] = np.nan
    out = tmp_path / "nan.ims"

    save_imaris(str(out), data, metadata={"scale": (1.0, 1.0, 1.0)})

    img, _ = load_image(str(out))
    try:
        np.testing.assert_array_equal(np.asarray(img[:, :, :, :, :]), data)
    finally:
        img.release()
    with h5py.File(out, "r") as f:
        path = "DataSet/ResolutionLevel 0/TimePoint 0/Channel 0"
        hist_max = float(_attr(f, path, "HistogramMax"))
    assert np.isclose(hist_max, np.nanmax(data[0, :, 0]), atol=1e-6)


def test_save_imaris_histogram_spans_data_range(tmp_path):
    """Bins must span [HistogramMin, HistogramMax], not the dtype's range
    (12-bit data in uint16 used to land in the first ~16 of 256 bins)."""
    data = np.zeros((1, 1, 1, 16, 16), dtype=np.uint16)
    data[..., :8] = 1000
    data[..., 8:] = 4000
    out = tmp_path / "hist.ims"

    save_imaris(str(out), data, metadata={"scale": (1.0, 1.0, 1.0)}, resolution_levels=False)

    with h5py.File(out, "r") as f:
        path = "DataSet/ResolutionLevel 0/TimePoint 0/Channel 0"
        hist = f[path + "/Histogram"][:]
        assert (_attr(f, path, "HistogramMin"), _attr(f, path, "HistogramMax")) == (
            "1000.000000",
            "4000.000000",
        )
    assert hist[0] == 128 and hist[-1] == 128


def test_save_imaris_keeps_frame_interval_without_timestamps(tmp_path):
    """Regression: with no timestamps (e.g. from a TIFF that only carries a
    frame interval) the writer stamped each timepoint with now(), losing
    the interval on reload."""
    from pyvistra.io import _extract_frame_interval_seconds, load_image

    data = np.zeros((3, 1, 1, 4, 4), dtype=np.uint16)
    out = tmp_path / "interval.ims"
    save_imaris(str(out), data, metadata={"scale": (1, 1, 1), "frame_interval_s": 30.0})

    img, meta = load_image(str(out))
    img.release()
    assert _extract_frame_interval_seconds(meta, 3) == 30.0


def test_save_imaris_unknown_timing_stays_unknown(tmp_path):
    from pyvistra.io import _extract_frame_interval_seconds, load_image

    out = tmp_path / "untimed.ims"
    save_imaris(str(out), np.zeros((3, 1, 1, 4, 4), np.uint16), metadata={"scale": (1, 1, 1)})

    img, meta = load_image(str(out))
    img.release()
    assert _extract_frame_interval_seconds(meta, 3) is None


def test_save_imaris_accepts_string_timestamps(tmp_path):
    """Regression: string timestamps (e.g. from JSON metadata) crashed on
    ``.strftime``."""
    from pyvistra.io import load_image

    out = tmp_path / "strings.ims"
    stamps = ["2026-01-01T00:00:00", "2026-01-01T00:00:05"]
    save_imaris(str(out), np.zeros((2, 1, 1, 4, 4), np.uint16), metadata={"timestamps": stamps})

    img, meta = load_image(str(out))
    img.release()
    assert [ts.isoformat() for ts in meta["timestamps"]] == stamps
