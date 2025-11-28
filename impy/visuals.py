import numpy as np
from vispy import scene
from vispy.color import Colormap


class CompositeImageVisual:
    """
    Manages multiple Vispy Image visuals to create a composite
    multi-channel rendering using additive blending.
    """

    def __init__(self, view, image_data):
        self.data = image_data
        self.view = view
        self.layers = []

        # State
        self.mode = "composite"
        self.active_channel_idx = 0

        self.current_slice_cache = None
        self.channel_clims = {}
        self.channel_gammas = {}
        self.channel_colors = [
            "#ffb100",
            "#49FF49",
            "#5BD6FF",
            "magenta",
            "cyan",
            "yellow",
        ]

        self._setup_layers()

    def _setup_layers(self):
        n_channels = self.data.shape[2]

        # 1. Determine decent default limits based on dtype
        # This prevents the "squashed handles" look on the slider
        dtype = self.data.dtype
        default_clim = (0, 255)

        if dtype == np.uint16:
            default_clim = (0, 65535)
        elif dtype.kind == "f":
            # For float, we assume 0.0-1.0 until we see data
            default_clim = (0.0, 1.0)

        for c in range(n_channels):
            if n_channels == 1:
                color_name = "white"
            else:
                color_name = self.channel_colors[c % len(self.channel_colors)]

            # Additive blending: Black -> Color
            cmap = Colormap(["black", color_name])

            image_visual = scene.visuals.Image(
                cmap=cmap,
                parent=self.view.scene,
                method="auto",
                interpolation="nearest",
            )

            # Force Additive Blending
            image_visual.set_gl_state(
                preset="translucent",
                blend=True,
                blend_func=("one", "one"),
                depth_test=False,
            )

            image_visual.order = -c
            self.layers.append(image_visual)

            self.channel_clims[c] = default_clim
            self.channel_gammas[c] = 1.0

    def set_mode(self, mode):
        self.mode = mode
        self._update_visibility()

    def set_active_channel(self, idx):
        self.active_channel_idx = idx
        self._update_visibility()

    def _update_visibility(self):
        for i, layer in enumerate(self.layers):
            if self.mode == "composite":
                layer.visible = True
                layer.set_gl_state(
                    blend=True,
                    blend_func=("one", "one"),
                    depth_test=False,
                )
            else:
                layer.visible = i == self.active_channel_idx

    def update_slice(self, t_idx, z_idx):
        try:
            volume_slice = self.data[t_idx, z_idx, :, :, :]
        except Exception as e:
            print(f"Error slicing data: {e}")
            return

        if volume_slice.ndim == 2:
            volume_slice = volume_slice[np.newaxis, :, :]

        self.current_slice_cache = volume_slice

        for c, layer in enumerate(self.layers):
            if c < volume_slice.shape[0]:
                plane = volume_slice[c]
                layer.set_data(plane)

                # Auto-Contrast Logic (Refinement)
                # If we are strictly at default (e.g. 0-65535) and the actual data
                # is tiny (e.g. max 400), we tighten it.
                current_clim = self.channel_clims[c]
                dtype = self.data.dtype

                # Check for "Suspiciously Default" limits vs "Real Data"
                if dtype == np.uint16 and current_clim == (0, 65535):
                    # Use percentiles for robustness against outliers
                    # Ignore zeros (background/padding) for more accurate contrast
                    valid_data = plane[plane > 0]
                    if valid_data.size > 0:
                        mn, mx = np.nanpercentile(valid_data, (0.1, 99.9))
                        # If data uses less than 10% of dynamic range, auto-scale
                        if mx < 6000:
                            self.set_clim(c, mn, max(mx, 1))

                elif dtype.kind == "f" and current_clim == (0.0, 1.0):
                    valid_data = plane[plane > 0]
                    if valid_data.size > 0:
                        mn, mx = np.nanpercentile(valid_data, (0.1, 99.9))
                        self.set_clim(c, mn, mx)

        self._update_visibility()

    def set_clim(self, channel_idx, vmin, vmax):
        if channel_idx < len(self.layers):
            self.layers[channel_idx].clim = (vmin, vmax)
            self.channel_clims[channel_idx] = (vmin, vmax)

    def get_clim(self, channel_idx):
        return self.channel_clims.get(channel_idx, (0, 255))

    def set_gamma(self, channel_idx, gamma):
        if channel_idx < len(self.layers):
            self.layers[channel_idx].gamma = gamma
            self.channel_gammas[channel_idx] = gamma

    def get_gamma(self, channel_idx):
        return self.channel_gammas.get(channel_idx, 1.0)

    def reset_camera(self, shape):
        _, _, _, Y, X = shape
        self.view.camera.rect = (0, 0, X, Y)
        self.view.camera.flip = (False, True, False)
