import numpy as np

from pyvistra.data.tracks import TrackTable, load_tracks_csv, save_tracks_csv


def test_track_table_sorts_by_track_and_time():
    tracks = TrackTable.from_arrays(
        track_id=[2, 1, 1, 2],
        t=[3, 5, 1, 2],
        x=[0, 1, 2, 3],
        y=[4, 5, 6, 7],
    )

    assert tracks.track_id.tolist() == [1, 1, 2, 2]
    assert tracks.t.tolist() == [1, 5, 2, 3]
    assert tracks.n_tracks == 2


def test_track_table_get_and_remove_track():
    tracks = TrackTable.from_arrays(
        track_id=[1, 1, 2, 2],
        t=[0, 1, 0, 1],
        x=[10, 11, 20, 21],
        y=[30, 31, 40, 41],
    )

    tr2 = tracks.get_track(2)
    assert tr2 is not None
    assert tr2["x"].tolist() == [20.0, 21.0]

    pruned = tracks.remove_track(1)
    assert pruned.n_tracks == 1
    assert np.all(pruned.track_id == 2)


def test_track_table_tracks_z_when_present():
    tracks = TrackTable.from_arrays(
        track_id=[1, 1, 1],
        t=[0, 1, 2],
        x=[0, 1, 2],
        y=[0, 1, 2],
        z=[4, 5, 6],
    )

    assert tracks.has_z
    tr = tracks.get_track(1)
    assert tr is not None
    assert tr["z"].tolist() == [4.0, 5.0, 6.0]


def test_track_csv_roundtrip_without_z(tmp_path):
    tracks = TrackTable.from_arrays(
        track_id=[1, 1, 2],
        t=[0, 1, 0],
        x=[10.5, 11.5, 20.5],
        y=[30.5, 31.5, 40.5],
    )

    path = tmp_path / "tracks.csv"
    save_tracks_csv(str(path), tracks)
    loaded = load_tracks_csv(str(path))

    assert loaded.n_rows == tracks.n_rows
    assert loaded.track_id.tolist() == tracks.track_id.tolist()
    assert loaded.t.tolist() == tracks.t.tolist()
    assert np.allclose(loaded.x, tracks.x)
    assert np.allclose(loaded.y, tracks.y)
    assert not loaded.has_z


def test_track_csv_roundtrip_with_z(tmp_path):
    tracks = TrackTable.from_arrays(
        track_id=[1, 1, 2],
        t=[0, 1, 0],
        x=[10.5, 11.5, 20.5],
        y=[30.5, 31.5, 40.5],
        z=[1.0, 2.0, 3.0],
    )

    path = tmp_path / "tracks_z.csv"
    save_tracks_csv(str(path), tracks)
    loaded = load_tracks_csv(str(path))

    assert loaded.has_z
    assert np.allclose(loaded.z, tracks.z)


def test_load_tracks_csv_missing_columns_raises(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("track_id,t,x\n1,0,1.0\n")

    try:
        load_tracks_csv(str(path))
    except ValueError as e:
        assert "Missing required columns" in str(e)
    else:
        raise AssertionError("Expected ValueError for missing columns")


def test_track_table_properties_sorted_and_validated():
    tracks = TrackTable.from_arrays(
        track_id=[2, 1],
        t=[0, 0],
        x=[10.0, 20.0],
        y=[30.0, 40.0],
        properties={"speed": [1.0, 2.0]},
    )

    # Sorted by (track_id, t): track 1 first, so its property follows.
    assert tracks.track_id.tolist() == [1, 2]
    assert tracks.properties["speed"].tolist() == [2.0, 1.0]

    try:
        TrackTable.from_arrays(
            track_id=[1, 2],
            t=[0, 0],
            x=[1.0, 2.0],
            y=[3.0, 4.0],
            properties={"speed": [1.0]},
        )
    except ValueError as e:
        assert "Property 'speed'" in str(e)
    else:
        raise AssertionError("Expected ValueError for mismatched property length")


def test_track_table_remove_track_slices_properties():
    tracks = TrackTable.from_arrays(
        track_id=[1, 1, 2, 2],
        t=[0, 1, 0, 1],
        x=[10, 11, 20, 21],
        y=[30, 31, 40, 41],
        properties={"quality": [0.1, 0.1, 0.9, 0.9]},
    )

    pruned = tracks.remove_track(1)
    assert pruned.properties["quality"].tolist() == [0.9, 0.9]


def test_track_csv_roundtrip_preserves_properties(tmp_path):
    tracks = TrackTable.from_arrays(
        track_id=[1, 1, 2],
        t=[0, 1, 0],
        x=[10.5, 11.5, 20.5],
        y=[30.5, 31.5, 40.5],
        properties={"speed": [1.0, 2.0, 3.0], "label": ["a", "b", "c"]},
    )

    path = tmp_path / "tracks_props.csv"
    save_tracks_csv(str(path), tracks)
    loaded = load_tracks_csv(str(path))

    assert np.allclose(loaded.properties["speed"], [1.0, 2.0, 3.0])
    assert loaded.properties["label"].tolist() == ["a", "b", "c"]


def test_aggregate_property_is_per_track_nanmean():
    tracks = TrackTable.from_arrays(
        track_id=[1, 1, 1, 2, 2, 3],
        t=[0, 1, 2, 0, 1, 0],
        x=[0] * 6,
        y=[0] * 6,
        properties={"v": [1.0, np.nan, 3.0, np.nan, np.nan, 7.0]},
    )
    out = tracks.aggregate_property(tracks.properties["v"])
    assert out[0] == 2.0
    assert np.isnan(out[1])  # all-NaN track: NaN, not 0 and no warning
    assert out[2] == 7.0


class _Recorder:
    """Stands in for a vispy visual: keeps the last set_data call."""

    order = 0
    visible = True

    def __init__(self, **_kw):
        self.last = None

    def set_data(self, **kw):
        self.last = kw

    def set_gl_state(self, *a, **k):
        pass


def _visual(monkeypatch, tracks, **style):
    from pyvistra.visuals import tracks as mod

    class _Scene:
        class visuals:
            Line = _Recorder
            Markers = _Recorder

    monkeypatch.setattr(mod, "scene", _Scene)

    class _View:
        scene = None

    visual = mod.TrackLayerVisual(_View(), **style)
    visual.set_tracks(tracks)
    return visual


def _two_tracks():
    # Track 1 lives frames 0-3, track 2 frames 5-7.
    return TrackTable.from_arrays(
        track_id=[1, 1, 1, 1, 2, 2, 2],
        t=[0, 1, 2, 3, 5, 6, 7],
        x=[0, 1, 2, 3, 10, 11, 12],
        y=[0, 0, 0, 0, 5, 5, 5],
        pixel_index_coordinates=False,
    )


def test_refresh_draws_each_tracks_trail_up_to_now(monkeypatch):
    visual = _visual(monkeypatch, _two_tracks(), trail_window=None)
    visual.set_time_z(2)
    pos = visual._line.last["pos"]
    # Track 1's first three vertices -> two segments; track 2 not born yet.
    assert pos.shape == (4, 2)
    assert pos[:, 0].tolist() == [0, 1, 1, 2]
    # Older segments fade: alpha rises with age within the visible trail.
    alpha = visual._line.last["color"][:, 3]
    assert alpha[0] < alpha[2]
    # Head only where the track has a vertex at the current frame.
    assert visual._heads.last["pos"].tolist() == [[2.0, 0.0]]


def test_refresh_trail_window_limits_history(monkeypatch):
    visual = _visual(monkeypatch, _two_tracks(), trail_window=1)
    visual.set_time_z(7)
    pos = visual._line.last["pos"]
    assert pos[:, 0].tolist() == [11, 12]  # frames 6-7 of track 2 only


def test_show_all_times_draws_every_track_whole(monkeypatch):
    visual = _visual(monkeypatch, _two_tracks(), show_all_times=True)
    visual.set_time_z(0)
    assert visual._line.last["pos"].shape == (2 * (3 + 2), 2)
    assert visual.style["show_all_times"] is True
