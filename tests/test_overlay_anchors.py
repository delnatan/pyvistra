"""Overlay corners are screen corners (y-down pixels), independent of camera flip."""

import pytest

from pyvistra.visuals.overlays import _anchor_point, _text_anchors


@pytest.mark.parametrize(
    "anchor, expected",
    [
        ("top-left", (10.0, 20.0)),
        ("top-right", (90.0, 20.0)),
        ("bottom-left", (10.0, 180.0)),
        ("bottom-right", (90.0, 180.0)),
    ],
)
def test_anchor_point_is_screen_corner(anchor, expected):
    assert _anchor_point(100, 200, anchor, 0.1, 0.1) == pytest.approx(expected)


def test_text_grows_into_viewport():
    # Right-anchored text must right-align or it runs off the edge; vispy's
    # anchor_y names are inverted in y-down space ("bottom" = glyphs below pos).
    assert _text_anchors("top-left") == ("left", "bottom")
    assert _text_anchors("bottom-right") == ("right", "top")
