"""The window's paint state (active mask layer, active label, brush size,
preserve labels) and the Label Manager's controls agree, whoever changes it."""

import numpy as np
import pytest
from qtpy.QtCore import Qt
from qtpy.QtTest import QTest

from pyvistra import imshow
from pyvistra.data.labels import SparseLabels
from pyvistra.ui.label_manager import LabelManager


@pytest.fixture
def win():
    window = imshow(np.zeros((1, 1, 1, 20, 20), np.uint16), "paint state")
    yield window
    window.close()


@pytest.fixture
def lm(win):
    manager = LabelManager()
    manager._connect_window(win)
    manager.set_active_window(win)
    yield manager
    manager._disconnect_window(win)
    manager.close()


def _labels(*ids):
    labels = SparseLabels((20, 20))
    for i in ids:
        labels.add(i, (np.array([i]), np.array([i])))
    return labels


def test_setters_signal_only_real_changes(win):
    seen = []
    win.paint_state_changed.connect(lambda: seen.append(1))
    win.active_label = 4
    win.active_label = 4
    win.brush_size = win.brush_size
    win.preserve_labels = not win.preserve_labels
    assert len(seen) == 2


def test_a_change_in_the_window_shows_in_the_label_manager(win, lm):
    win.add_mask_layer("m", _labels(1, 7))
    win.active_label = 7
    win.brush_size = 11
    win.preserve_labels = False
    assert lm.active_label_spin.value() == 7
    assert lm.brush_size_spin.value() == 11
    assert not lm.preserve_checkbox.isChecked()
    assert [it.data(Qt.UserRole) for it in lm.label_list.selectedItems()] == [7]


def test_a_keyboard_shortcut_shows_in_the_label_manager(win, lm):
    win.add_mask_layer("m", _labels(1, 2))
    QTest.keyClick(win, Qt.Key_N)  # next free label
    assert win.active_label == 3 and lm.active_label_spin.value() == 3
    QTest.keyClick(win, Qt.Key_BracketRight)
    assert lm.brush_size_spin.value() == win.brush_size


def test_the_label_manager_sets_the_window_through_its_setters(win, lm):
    seen = []
    win.paint_state_changed.connect(lambda: seen.append(1))
    lm.active_label_spin.setValue(9)
    lm.brush_size_spin.setValue(3)
    assert win.active_label == 9 and win.brush_size == 3 and len(seen) == 2


def test_switching_the_active_mask_layer_lists_its_labels(win, lm):
    win.add_mask_layer("a", _labels(1))
    win.add_mask_layer("b", _labels(4, 5))
    listed = lambda: sorted(lm.label_list.item(r).data(Qt.UserRole) for r in range(lm.label_list.count()))  # noqa: E731
    assert listed() == [1]
    win.set_active_mask_layer("b")
    assert listed() == [4, 5]
    win.remove_mask_layer("b")
    assert listed() == [1]


def test_a_merge_selection_survives_picking_its_rows(win, lm):
    win.add_mask_layer("m", _labels(1, 2, 3))
    items = {lm.label_list.item(r).data(Qt.UserRole): lm.label_list.item(r) for r in range(3)}
    items[1].setSelected(True)
    items[3].setSelected(True)
    lm.on_item_clicked(items[3])  # picking a selected row makes it the active label
    assert win.active_label == 3
    assert sorted(it.data(Qt.UserRole) for it in lm.label_list.selectedItems()) == [1, 3]
