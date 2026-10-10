"""Label painting in ImageWindow (brush/contour fill into SparseLabels)."""

import numpy as np

from pyvistra import imshow
from pyvistra.data.labels import SparseLabels


def test_propagated_fill_preserves_labels_on_other_slices():
    """Regression: with "preserve labels" on, a Z-propagated contour fill
    only checked occupancy on the displayed slice, so it claimed pixels
    other labels owned on every other slice."""
    labels = SparseLabels((3, 20, 20))
    labels.add(1, (np.array([2]), np.array([10]), np.array([10])))  # z=2 only
    win = imshow(np.zeros((1, 3, 1, 20, 20), np.uint16), "paint")
    try:
        win.add_mask_layer("m", labels)
        win.z_idx = 0
        win.active_label = 2
        win.brush_size = 1
        win.preserve_labels = True
        win._mask_propagate_z = True
        # Closed square contour around (10, 10).
        win._stroke_points = [(x, 5.0) for x in range(5, 16)] + [
            (15.0, y) for y in range(5, 16)
        ] + [(x, 15.0) for x in range(15, 4, -1)] + [(5.0, y) for y in range(15, 4, -1)]
        win._fill_closed_contour()

        filled = set(zip(*labels.coords(2)))
        assert (0, 10, 10) in filled and (1, 10, 10) in filled
        assert (2, 10, 10) not in filled  # label 1's pixel stays label 1's
        assert labels.to_dense()[2, 10, 10] == 1
    finally:
        win.close()
