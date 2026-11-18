from __future__ import annotations

import colorsys

import numpy as np
from vispy import scene

from .. import colormaps as cmap
from ..data.property_filter import (
    PropertyFilterSpec,
    PropertyRange,
    SelectionEffect,
    compute_mask,
    resolve_row_effects,
)
from ..data.tracks import TrackTable

DEFAULT_STYLE = {
    "trail_window": 30,
    "line_width": 2.0,
    "opacity": 0.9,
    "color_mode": "auto",
    "fixed_color": (1.0, 1.0, 1.0, 1.0),
    "color_property": None,
    "colormap_name": "viridis",
    "color_clim": None,
    "show_heads": True,
    "head_size": 8.0,
    "min_track_length": 1,
    "show_all_times": False,
}

_TRACK_LENGTH_PROPERTY = "__track_length__"


class TrackLayerVisual:
    """Renders time-aware polylines for tracking overlays."""

    def __init__(
        self,
        view,
        *,
        trail_window=30,
        line_width=2.0,
        opacity=0.9,
        color_mode="auto",
        fixed_color=(1.0, 1.0, 1.0, 1.0),
        color_property=None,
        colormap_name="viridis",
        color_clim=None,
        show_heads=True,
        head_size=8.0,
        min_track_length=1,
        show_all_times=False,
    ):
        self.view = view
        self._tracks: TrackTable | None = None
        self._visible = True
        self._trail_window = trail_window
        self._line_width = float(line_width)
        self._opacity = float(opacity)
        self._color_mode = str(color_mode)
        self._fixed_color = tuple(fixed_color)
        self._color_property = color_property
        self._colormap_name = str(colormap_name)
        self._color_clim = None if color_clim is None else tuple(color_clim)
        self._show_heads = bool(show_heads)
        self._head_size = float(head_size)
        self._min_track_length = int(min_track_length)
        # Draw every track whole regardless of the current frame (a map
        # of where each trajectory went), instead of its trail up to now.
        self._show_all_times = bool(show_all_times)
        self._selection_spec = PropertyFilterSpec()
        self._selection_effect = SelectionEffect()
        self._selection_colors: dict[int, tuple[float, float, float, float] | None] = {}
        self._t_idx = 0
        self._z_idx = 0

        self._line = scene.visuals.Line(
            pos=np.zeros((0, 2), dtype=np.float32),
            color=(1.0, 1.0, 1.0, 0.0),
            connect="segments",
            width=self._line_width,
            parent=self.view.scene,
        )
        self._heads = scene.visuals.Markers(parent=self.view.scene)

        self._line.order = 5_000
        self._heads.order = 5_001

        self._line.set_gl_state(
            preset="translucent",
            blend=True,
            blend_func=("src_alpha", "one_minus_src_alpha"),
            depth_test=False,
        )
        self._heads.set_gl_state(
            preset="translucent",
            blend=True,
            blend_func=("src_alpha", "one_minus_src_alpha"),
            depth_test=False,
        )

    def _clear_line(self):
        # VisPy Line can fail on empty per-vertex color arrays; use a scalar RGBA.
        self._line.set_data(
            pos=np.zeros((0, 2), dtype=np.float32),
            color=(1.0, 1.0, 1.0, 0.0),
            connect="segments",
            width=self._line_width,
        )

    @property
    def visible(self):
        return self._visible

    @visible.setter
    def visible(self, value):
        self._visible = bool(value)
        self._line.visible = self._visible
        self._heads.visible = self._visible

    @property
    def style(self) -> dict[str, object]:
        return {
            "trail_window": self._trail_window,
            "line_width": self._line_width,
            "opacity": self._opacity,
            "color_mode": self._color_mode,
            "fixed_color": self._fixed_color,
            "color_property": self._color_property,
            "colormap_name": self._colormap_name,
            "color_clim": self._color_clim,
            "show_heads": self._show_heads,
            "head_size": self._head_size,
            "min_track_length": self._min_track_length,
            "show_all_times": self._show_all_times,
        }

    def set_tracks(self, tracks: TrackTable | None):
        self._tracks = tracks
        self.refresh()

    def set_property_selection(self, spec: PropertyFilterSpec, effect: SelectionEffect):
        self._selection_spec = spec
        self._selection_effect = effect
        self.refresh()

    def get_property_selection(self) -> tuple[PropertyFilterSpec, SelectionEffect]:
        return self._selection_spec, self._selection_effect

    def set_time_z(self, t_idx: int, z_idx: int = 0):
        self._t_idx = int(t_idx)
        self._z_idx = int(z_idx)
        self.refresh()

    def set_trail_window(self, trail_window: int | None):
        self._trail_window = trail_window
        self.refresh()

    def set_line_width(self, line_width: float):
        self._line_width = float(line_width)
        self.refresh()

    def set_opacity(self, opacity: float):
        self._opacity = float(np.clip(opacity, 0.0, 1.0))
        self.refresh()

    def set_style(self, **kwargs):
        if "trail_window" in kwargs:
            self._trail_window = kwargs["trail_window"]
        if "line_width" in kwargs:
            self._line_width = float(kwargs["line_width"])
        if "opacity" in kwargs:
            self._opacity = float(np.clip(kwargs["opacity"], 0.0, 1.0))
        if "color_mode" in kwargs:
            self._color_mode = str(kwargs["color_mode"])
        if "fixed_color" in kwargs:
            self._fixed_color = tuple(kwargs["fixed_color"])
        if "color_property" in kwargs:
            self._color_property = kwargs["color_property"]
        if "colormap_name" in kwargs:
            self._colormap_name = str(kwargs["colormap_name"])
        if "color_clim" in kwargs:
            clim = kwargs["color_clim"]
            self._color_clim = None if clim is None else tuple(clim)
        if "show_heads" in kwargs:
            self._show_heads = bool(kwargs["show_heads"])
        if "head_size" in kwargs:
            self._head_size = float(kwargs["head_size"])
        if "min_track_length" in kwargs:
            self._min_track_length = int(kwargs["min_track_length"])
        if "show_all_times" in kwargs:
            self._show_all_times = bool(kwargs["show_all_times"])
        self.refresh()

    def remove(self):
        self._line.parent = None
        self._heads.parent = None

    def refresh(self):
        """Redraw the trails at the current (t, z).

        Vectorized over every vertex at once -- this runs on every frame
        change, and a per-segment Python loop made scrubbing a movie with
        tens of thousands of tracks stall. Rows are sorted by (track, t)
        (a TrackTable invariant), so a track's visible vertices are one
        contiguous run of the visible-row index, and consecutive visible
        rows of the same track are exactly the segments to draw.
        """
        tracks = self._tracks
        if not self._visible or tracks is None or tracks.n_rows == 0:
            self._clear_line()
            self._heads.set_data(pos=np.zeros((0, 2), dtype=np.float32), size=0)
            return

        track_visible = self._build_track_visibility()
        track_colors = self._build_track_colors()
        track_ids = tracks.track_ids
        visible_by_track = np.array([track_visible[int(tid)] for tid in track_ids], dtype=bool)
        rgb_by_track = np.array(
            [track_colors.get(int(tid), (1.0, 1.0, 1.0))[:3] for tid in track_ids], dtype=np.float32
        ).reshape(-1, 3)

        # Track index (into track_ids) of every row.
        row_track = np.repeat(np.arange(len(track_ids)), tracks._track_counts)

        # pixel_index_coordinates (declared on the TrackTable itself, by
        # whoever loaded/constructed it — see TrackTable's docstring): when
        # set, x/y are array-index coordinates and need +0.5 to land on
        # vispy Image's pixel-center convention (matches the ortho-view
        # crosshair in viewers/ortho.py).
        offset = 0.5 if tracks.pixel_index_coordinates else 0.0

        t = tracks.t
        shown = visible_by_track[row_track]
        if not self._show_all_times:
            shown &= t <= self._t_idx
            if self._trail_window is not None:
                shown &= t >= (self._t_idx - int(self._trail_window))
        if tracks.has_z:
            # 2D canvas display: only show points close to current z slice.
            shown &= np.abs(tracks.z - self._z_idx) <= 0.5

        rows = np.flatnonzero(shown)
        if rows.size == 0:
            self._clear_line()
            self._heads.set_data(pos=np.zeros((0, 2), dtype=np.float32), size=0)
            return

        trk = row_track[rows]
        tt = t[rows]
        pos = np.column_stack((tracks.x[rows] + offset, tracks.y[rows] + offset)).astype(np.float32)

        # Each track's run within `rows`: its first and last visible row.
        run_start = np.flatnonzero(np.r_[True, trk[1:] != trk[:-1]])
        run_end = np.r_[run_start[1:], rows.size] - 1
        run_of_row = np.repeat(np.arange(run_start.size), run_end - run_start + 1)

        a = np.flatnonzero(trk[:-1] == trk[1:])  # segment i joins visible rows a, a+1
        if a.size:
            run = run_of_row[a]
            t_first = tt[run_start[run]].astype(np.float64)
            span = tt[run_end[run]].astype(np.float64) - t_first
            age = np.ones(a.size, dtype=np.float64)
            has_span = span > 0
            age[has_span] = (tt[a + 1][has_span] - t_first[has_span]) / span[has_span]
            alpha = ((0.2 + 0.8 * age) * self._opacity).astype(np.float32)
            colors = np.column_stack((rgb_by_track[trk[a]], alpha))
            seg_pos = np.empty((2 * a.size, 2), dtype=np.float32)
            seg_pos[0::2] = pos[a]
            seg_pos[1::2] = pos[a + 1]
            self._line.set_data(
                pos=seg_pos,
                color=np.repeat(colors, 2, axis=0).astype(np.float32),
                connect="segments",
                width=self._line_width,
            )
        else:
            self._clear_line()

        # Only draw a head when the track actually has a point at the
        # current frame — otherwise a track that ended earlier keeps
        # showing a stale, frozen head for the rest of the trail window.
        heads = run_end[tt[run_end] == self._t_idx] if self._show_heads else run_end[:0]
        if heads.size:
            head_colors = np.column_stack(
                (rgb_by_track[trk[heads]], np.full(heads.size, self._opacity, dtype=np.float32))
            )
            self._heads.set_data(
                pos=pos[heads],
                face_color=head_colors.astype(np.float32),
                edge_color=(1.0, 1.0, 1.0, min(1.0, self._opacity + 0.1)),
                edge_width=1.0,
                symbol="disc",
                size=self._head_size,
            )
        else:
            self._heads.set_data(pos=np.zeros((0, 2), dtype=np.float32), size=0)

    def _track_scalar(self, prop: np.ndarray) -> np.ndarray:
        """Per-track nanmean of a per-row property array, aligned with track_ids order."""
        return self._tracks.aggregate_property(prop)

    def _build_track_visibility(self) -> dict[int, bool]:
        """Resolve one visibility flag per track_id from two independent,
        AND-combined gates: min_track_length (a dedicated, track-length-
        specific concept) and the generic property selection's hide effect.

        Also stashes `self._selection_colors` (per-track highlight color, or
        None) for `_build_track_colors` to apply — computed here so the
        per-track property aggregation (`_track_scalar`) only runs once.
        """
        tracks = self._tracks
        track_ids = tracks.track_ids
        n_tracks = len(track_ids)

        track_lengths = np.array(
            [sl.stop - sl.start for _, sl in tracks.iter_track_slices()],
            dtype=np.float64,
        )
        length_spec = PropertyFilterSpec(
            ranges=(PropertyRange(_TRACK_LENGTH_PROPERTY, min_value=self._min_track_length),)
        )
        length_mask = compute_mask(
            n_tracks, {_TRACK_LENGTH_PROPERTY: track_lengths}, length_spec
        )

        aggregated: dict[str, np.ndarray] = {
            rng.name: self._track_scalar(tracks.properties[rng.name])
            for rng in self._selection_spec.ranges
            if rng.name in tracks.properties
        }
        sel_visible, sel_color = resolve_row_effects(
            n_tracks, aggregated, self._selection_spec, self._selection_effect
        )

        self._selection_colors = {
            int(tid): (
                None
                if sel_color is None or np.isnan(sel_color[i]).any()
                else tuple(sel_color[i].tolist())
            )
            for i, tid in enumerate(track_ids)
        }

        return {
            int(tid): bool(length_mask[i] and sel_visible[i])
            for i, tid in enumerate(track_ids)
        }

    def _build_track_colors(self) -> dict[int, tuple[float, float, float]]:
        """Resolve one RGB color per track_id under the current color mode.

        "property" colors by the per-track mean of a properties column
        (one color per whole trajectory, not per-vertex/time-varying) —
        falls back to "auto" if the property isn't present on the table.

        A selection highlight color (`self._selection_colors`, populated by
        `_build_track_visibility`, which must run first each refresh) always
        overrides the color_mode result — an explicit highlight action
        should be visually unambiguous regardless of the active color mode.
        """
        tracks = self._tracks

        if self._color_mode == "fixed":
            rgb = tuple(float(c) for c in self._fixed_color[:3])
            base = {int(tid): rgb for tid in tracks.track_ids}
        elif (
            self._color_mode == "property"
            and self._color_property is not None
            and self._color_property in tracks.properties
        ):
            prop = tracks.properties[self._color_property]
            vals_arr = self._track_scalar(prop)
            means = {int(tid): float(vals_arr[i]) for i, tid in enumerate(tracks.track_ids)}

            if self._color_clim is not None:
                lo, hi = self._color_clim
            elif means:
                vals = np.array(list(means.values()), dtype=np.float64)
                lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
            else:
                lo, hi = 0.0, 1.0
            span = (hi - lo) if hi > lo else 1.0

            colormap, _ = cmap.get(self._colormap_name)
            tids = list(means.keys())
            norm = np.clip(
                (np.array([means[t] for t in tids], dtype=np.float64) - lo) / span,
                0.0, 1.0,
            )
            rgba = colormap.map(norm)
            base = {int(tid): tuple(rgba[i, :3]) for i, tid in enumerate(tids)}
        else:
            # "auto" (or "property" mode with an unset/missing property column):
            # deterministic hue-by-track-id, unchanged from the original default.
            base = {
                int(tid): colorsys.hsv_to_rgb((int(tid) * 0.61803398875) % 1.0, 0.8, 0.95)
                for tid in tracks.track_ids
            }

        return {
            tid: (
                self._selection_colors[tid][:3]
                if self._selection_colors.get(tid) is not None
                else rgb
            )
            for tid, rgb in base.items()
        }
