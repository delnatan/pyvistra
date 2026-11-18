"""ImageWindow.rename_layer and replacing a layer's data: the window's
per-type registries, the unified LayerList and the visuals stay in step."""

import numpy as np
import pytest

from pyvistra import imshow
from pyvistra.data.labels import SparseLabels
from pyvistra.data.point_commands import AddPoint
from pyvistra.data.points import PointTable
from pyvistra.data.tracks import TrackTable


@pytest.fixture
def win():
    window = imshow(np.zeros((3, 1, 1, 20, 20), np.uint16), "rename")
    yield window
    window.close()


def _tracks():
    return TrackTable.from_arrays(track_id=[1, 1], t=[0, 1], x=[2, 3], y=[2, 2])


def test_rename_moves_every_layer_type_and_keeps_it_working(win):
    win.add_mask_layer("mask")
    win.add_track_layer("tr", _tracks())
    win.add_point_layer("pts", PointTable.from_arrays(x=[1.0], y=[1.0]))
    win.add_shape_layer("shapes")
    renamed = []
    win.layer_renamed.connect(lambda old, new: renamed.append((old, new)))
    for old, new in (("mask", "cells"), ("tr", "tracks"), ("pts", "spots"), ("shapes", "rois")):
        win.rename_layer(old, new)
        assert new in win.layers and old not in win.layers
    assert renamed == [("mask", "cells"), ("tr", "tracks"), ("pts", "spots"), ("shapes", "rois")]
    assert list(win._mask_layers) == ["cells"] and win._active_mask_layer == "cells"
    assert list(win._track_layers) == ["tracks"] and win._active_track_layer == "tracks"
    assert list(win._point_layers) == ["spots"] and win._active_point_layer == "spots"
    # The renamed layers still answer to their new names...
    win.set_track_layer_style("tracks", line_width=3.0)
    assert win._track_layers["tracks"]["style"]["line_width"] == 3.0
    win.remove_mask_layer("cells")
    assert "cells" not in win.layers and not win._mask_layers


def test_point_edits_still_reach_a_renamed_layer(win):
    win.add_point_layer("pts", PointTable.from_arrays(x=[1.0], y=[1.0]))
    win.rename_layer("pts", "spots")
    layer = win.layers["spots"]
    layer.undo_stack.push(AddPoint(x=5.0, y=6.0, t=0), layer.data)
    assert win._point_layers["spots"]["points"].n_rows == 2


def test_rename_refuses_before_changing_anything(win):
    win.add_mask_layer("a")
    win.add_mask_layer("b")
    with pytest.raises(ValueError):
        win.rename_layer("a", "b")
    with pytest.raises(KeyError):
        win.rename_layer("nope", "c")
    assert list(win._mask_layers) == ["a", "b"] and win.layers.names()[-2:] == ["a", "b"]


def test_rename_keeps_the_registry_order(win):
    for name in ("a", "b", "c"):
        win.add_mask_layer(name)
    win.rename_layer("b", "middle")
    assert list(win._mask_layers) == ["a", "middle", "c"]


def test_replacing_mask_labels_keeps_layer_data_and_paint_target_in_step(win):
    win.add_mask_layer("m")
    replacement = SparseLabels((20, 20))
    changed = []
    win.label_changed.connect(changed.append)
    win.set_mask_layer_labels("m", replacement)
    assert win.layers["m"].data is replacement
    assert win.labels is replacement
    assert changed == [replacement]
    # The window-level setter takes the same path.
    other = SparseLabels((20, 20))
    win.labels = other
    assert win.layers["m"].data is other and win._mask_layers["m"]["labels"] is other


def test_painting_a_renamed_mask_layer(win):
    win.add_mask_layer("m")
    win.rename_layer("m", "regions")
    win.set_active_mask_layer("regions")
    win.active_label = 3
    win.brush_size = 1
    win.preserve_labels = False
    win._paint_stroke(10.0, 10.0)
    assert 3 in win.layers["regions"].data


def test_replacing_points_reaches_the_undoable_holder(win):
    win.add_point_layer("pts", PointTable.from_arrays(x=[1.0], y=[1.0]))
    replacement = PointTable.from_arrays(x=[4.0, 5.0], y=[4.0, 5.0])
    win.set_point_layer_points("pts", replacement)
    layer = win.layers["pts"]
    assert layer.data.table is replacement
    # An edit after the replacement builds on it, not on the old table.
    layer.undo_stack.push(AddPoint(x=9.0, y=9.0, t=0), layer.data)
    assert win._point_layers["pts"]["points"].n_rows == 3


def test_layer_manager_rename_goes_through_the_window(win, monkeypatch):
    from qtpy.QtWidgets import QInputDialog

    from pyvistra.ui.layer_manager import LayerManager

    win.add_mask_layer("m")
    manager = LayerManager()
    manager._current_window = win
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("cells", True)))
    manager._rename_layer("m")
    assert "cells" in win.layers and list(win._mask_layers) == ["cells"]
    manager.close()
