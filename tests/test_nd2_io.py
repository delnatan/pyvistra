import numpy as np
import tifffile

from pyvistra.io import (
    _extract_nd2_channels,
    _normalize_nd2_to_5d,
    load_image,
    save_tiff,
)


def test_normalize_nd2_to_5d_reorders_tczyx():
    data = np.zeros((2, 3, 4, 5, 6), dtype=np.uint16)  # T, C, Z, Y, X
    sizes = {"T": 2, "C": 3, "Z": 4, "Y": 5, "X": 6}

    out, collapsed = _normalize_nd2_to_5d(data, sizes)

    assert out.shape == (2, 4, 3, 5, 6)
    assert collapsed == []


def test_normalize_nd2_to_5d_collapses_extra_dims_into_t():
    data = np.zeros((2, 3, 4, 5, 6, 7), dtype=np.uint16)  # P, T, Z, C, Y, X
    sizes = {"P": 2, "T": 3, "Z": 4, "C": 5, "Y": 6, "X": 7}

    out, collapsed = _normalize_nd2_to_5d(data, sizes)

    assert out.shape == (6, 4, 5, 6, 7)
    assert collapsed == ["P"]


def test_extract_nd2_channels_maps_common_fields():
    metadata = {
        "channels": [
            {
                "channel": {
                    "name": "GFP",
                    "emissionLambdaNm": 510.0,
                    "excitationLambdaNm": 488.0,
                },
                "time": {"exposureTimeMs": 25.0},
            },
            {},
        ]
    }

    channels = _extract_nd2_channels(metadata, n_channels=2)

    assert channels[0]["name"] == "GFP"
    assert channels[0]["emission_wavelength"] == 510.0
    assert channels[0]["excitation_wavelength"] == 488.0
    assert channels[0]["exposure_time"] == 0.025
    assert channels[1]["name"] == "Channel 1"


def test_save_tiff_writes_imagej_frame_interval_from_timestamp_seconds(
    tmp_path,
):
    data = np.zeros((4, 1, 1, 8, 8), dtype=np.uint16)
    meta = {"timestamp_seconds": [0.0, 2.0, 4.0, 6.0]}
    out = tmp_path / "timelapse.tif"

    save_tiff(str(out), data, metadata=meta)

    with tifffile.TiffFile(str(out)) as tif:
        ij = tif.imagej_metadata or {}

    assert ij.get("finterval") == 2.0
    assert ij.get("tunit") == "sec"
    assert ij.get("fps") == 0.5


def test_load_tiff_reads_imagej_frame_interval_seconds(tmp_path):
    data = np.zeros((3, 1, 1, 8, 8), dtype=np.uint16)
    out = tmp_path / "timelapse_roundtrip.tif"

    save_tiff(
        str(out),
        data,
        metadata={"timestamp_seconds": [0.0, 1.5, 3.0]},
    )
    _, meta = load_image(str(out), use_memmap=False)

    assert meta["frame_interval_s"] == 1.5


def test_save_tiff_labels_frequency_space_scale_correctly(tmp_path):
    """FFT output tags itself space="frequency" (see fft_dialog.py) --
    saving it to file must not silently claim its cycles/um scale is a
    real-space "um" pixel size.
    """
    data = np.zeros((1, 1, 1, 8, 8), dtype=np.float32)
    out = tmp_path / "fft_result.tif"

    save_tiff(
        str(out),
        data,
        scale=(1.0, 0.1, 0.1),
        metadata={"space": "frequency"},
    )

    with tifffile.TiffFile(str(out)) as tif:
        ij = tif.imagej_metadata or {}

    assert ij.get("unit") == "1/um"


def test_save_tiff_labels_real_space_scale_as_um(tmp_path):
    data = np.zeros((1, 1, 1, 8, 8), dtype=np.float32)
    out = tmp_path / "real_space.tif"

    save_tiff(str(out), data, scale=(1.0, 0.1, 0.1), metadata={})

    with tifffile.TiffFile(str(out)) as tif:
        ij = tif.imagej_metadata or {}

    assert ij.get("unit") == "um"


class _FakeND2:
    """Minimal stand-in for nd2.ND2File's per-frame time API."""

    def __init__(self, sizes, rel_ms):
        loops = [k for k in sizes if k in ("P", "T", "Z")]
        import itertools

        self.loop_indices = [
            dict(zip(loops, idx))
            for idx in itertools.product(*(range(sizes[k]) for k in loops))
        ]
        self._rel_ms = rel_ms

    def frame_metadata(self, seq_idx):
        ms = self._rel_ms(self.loop_indices[seq_idx])
        return {"channels": [{"time": {"relativeTimeMs": ms}}]}


def test_nd2_timestamps_follow_collapsed_position_loop():
    """Regression: with a P loop folded into T, timestamps were keyed by the
    raw "T" loop index -- position 1's frames got position 0's slots and
    every slot past n_T stayed empty."""
    from pyvistra.io import _extract_nd2_timestamp_seconds

    sizes = {"P": 2, "T": 3, "Z": 2, "C": 1, "Y": 4, "X": 4}
    # Each (p, t) acquired at t*10 s + p*1 s; Z frames a few ms later.
    fake = _FakeND2(sizes, lambda i: (i["T"] * 10 + i["P"]) * 1000 + i["Z"] * 5)

    secs = _extract_nd2_timestamp_seconds(fake, sizes)

    # pyvistra T index = p * n_T + t (same order as _normalize_nd2_to_5d).
    assert secs == [0.0, 10.0, 20.0, 1.0, 11.0, 21.0]
    data = np.zeros((2, 3, 2, 1, 4, 4), np.uint16)
    out, _ = _normalize_nd2_to_5d(data, sizes)
    assert out.shape[0] == len(secs)
