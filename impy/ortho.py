import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QMainWindow,
    QWidget,
    QGridLayout,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QComboBox,
    QAction,
)
from vispy import scene
from vispy.visuals.transforms import STTransform

from .visuals import CompositeImageVisual
from .widgets import ContrastDialog, MetadataDialog
from .manager import manager

class OrthoVisualProxy:
    """
    Proxies calls to multiple CompositeImageVisual instances to keep them in sync.
    Used by ContrastDialog to control all 3 views simultaneously.
    """
    def __init__(self, visuals):
        self.visuals = visuals # [yx, zy, zx]
        # We treat the first visual (YX) as the "primary" for reading state
        self.primary = visuals[0]

    @property
    def layers(self):
        # Return layers of primary so dialog can iterate them
        return self.primary.layers

    @property
    def current_slice_cache(self):
        return self.primary.current_slice_cache
    
    @property
    def channel_colors(self):
        return self.primary.channel_colors

    def set_clim(self, channel_idx, vmin, vmax):
        for v in self.visuals:
            v.set_clim(channel_idx, vmin, vmax)

    def get_clim(self, channel_idx):
        return self.primary.get_clim(channel_idx)

    def set_gamma(self, channel_idx, gamma):
        for v in self.visuals:
            v.set_gamma(channel_idx, gamma)

    def get_gamma(self, channel_idx):
        return self.primary.get_gamma(channel_idx)
    
    def set_mode(self, mode):
        for v in self.visuals:
            v.set_mode(mode)
            
    def set_active_channel(self, idx):
        for v in self.visuals:
            v.set_active_channel(idx)


class TransposedProxy:
    """
    Wraps a 5D data object and presents a transposed view.
    Handles mapping slices to the original data and transposing the result.
    """
    def __init__(self, data, perm):
        self.data = data
        self.perm = perm # e.g. (0, 4, 2, 3, 1)
        
        # Calculate new shape
        self.shape = tuple(data.shape[p] for p in perm)
        self.dtype = data.dtype
        self.ndim = 5

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        
        # Pad key
        if len(key) < 5:
            key = key + (slice(None),) * (5 - len(key))

        # 1. Map key to original dimensions
        original_key = [slice(None)] * 5
        for i, p in enumerate(self.perm):
            original_key[p] = key[i]
        original_key = tuple(original_key)

        # 2. Get data
        res = self.data[original_key]

        # 3. Transpose result if necessary
        # Identify which dimensions remain
        # The result 'res' will have dimensions corresponding to those 
        # indices in 'original_key' that were slices (not integers).
        # They will appear in the order of the original data (0..4).
        
        # We want them in the order they appeared in 'key'.
        
        # Indices in 'key' that are slices
        kept_dims_view = [i for i, k in enumerate(key) if isinstance(k, slice)]
        
        # Corresponding indices in 'original data'
        target_dims = [self.perm[i] for i in kept_dims_view]
        
        # The actual dimensions present in 'res' are 'target_dims' sorted.
        present_dims = sorted(target_dims)
        
        # Map from original dim index to position in 'res'
        dim_to_pos = {d: i for i, d in enumerate(present_dims)}
        
        # Construct permutation for 'res'
        # We want the k-th dimension of output to correspond to target_dims[k]
        res_perm = [dim_to_pos[d] for d in target_dims]
        
        # Only transpose if permutation is not identity
        if res_perm != list(range(len(res_perm))):
             res = res.transpose(res_perm)
             
        return res


class OrthoViewer(QMainWindow):
    def __init__(self, data, meta=None, title="Ortho View"):
        super().__init__()
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle(title)
        self.resize(1000, 800)

        self.data = data # (T, Z, C, Y, X)
        self.meta = meta or {}
        self.T, self.Z, self.C, self.Y, self.X = self.data.shape
        
        # Scale (z, y, x)
        self.scale = self.meta.get("scale", (1.0, 1.0, 1.0))
        sz, sy, sx = self.scale

        # Current Position (in pixels)
        self.cz = self.Z // 2
        self.cy = self.Y // 2
        self.cx = self.X // 2
        self.ct = 0

        # Register
        self.window_id = manager.register(self)

        # -- Layout --
        central = QWidget()
        self.setCentralWidget(central)
        self.main_layout = QVBoxLayout(central)
        
        # Grid for Views
        self.grid = QGridLayout()
        self.grid.setSpacing(2)
        self.main_layout.addLayout(self.grid, 1)

        # -- 1. Create Canvases --
        # YX View (Top-Left)
        self.canvas_yx = scene.SceneCanvas(keys=None, bgcolor="black")
        self.view_yx = self.canvas_yx.central_widget.add_view()
        self.view_yx.camera = "panzoom"
        self.view_yx.camera.aspect = 1
        self.grid.addWidget(self.canvas_yx.native, 0, 0)

        # ZY View (Top-Right) - Rotated: Y vertical, Z horizontal
        self.canvas_zy = scene.SceneCanvas(keys=None, bgcolor="black")
        self.view_zy = self.canvas_zy.central_widget.add_view()
        self.view_zy.camera = "panzoom"
        self.view_zy.camera.aspect = 1
        self.grid.addWidget(self.canvas_zy.native, 0, 1)

        # ZX View (Bottom-Left) - Rotated: Z vertical, X horizontal
        self.canvas_zx = scene.SceneCanvas(keys=None, bgcolor="black")
        self.view_zx = self.canvas_zx.central_widget.add_view()
        self.view_zx.camera = "panzoom"
        self.view_zx.camera.aspect = 1
        self.grid.addWidget(self.canvas_zx.native, 1, 0)
        
        # -- 2. Create Visuals --
        # YX: (T, Z, C, Y, X) -> Slice Z -> (C, Y, X)
        self.vis_yx = CompositeImageVisual(self.view_yx, self.data, scale=(sy, sx))
        
        # ZY: Need (T, X, C, Y, Z). Slice X -> (C, Y, Z)
        # Transpose data: (0, 4, 2, 3, 1) -> T, X, C, Y, Z
        data_zy = TransposedProxy(self.data, (0, 4, 2, 3, 1))
        self.vis_zy = CompositeImageVisual(self.view_zy, data_zy, scale=(sy, sz))
        
        # ZX: Need (T, Y, C, Z, X). Slice Y -> (C, Z, X)
        # Transpose data: (0, 3, 2, 1, 4) -> T, Y, C, Z, X
        data_zx = TransposedProxy(self.data, (0, 3, 2, 1, 4))
        self.vis_zx = CompositeImageVisual(self.view_zx, data_zx, scale=(sz, sx))

        self.proxy = OrthoVisualProxy([self.vis_yx, self.vis_zy, self.vis_zx])

        # -- 3. Crosshairs --
        # YX View: V-Line at X, H-Line at Y
        self.line_yx_v = scene.visuals.InfiniteLine(pos=self.cx, color=(1, 1, 0, 0.5), vertical=True, parent=self.view_yx.scene)
        self.line_yx_h = scene.visuals.InfiniteLine(pos=self.cy, color=(1, 1, 0, 0.5), vertical=False, parent=self.view_yx.scene)
        self.line_yx_v.set_gl_state(
            preset="translucent",
            blend=True,
            blend_func=("src_alpha", "one_minus_src_alpha"),
            depth_test=False
        )
        self.line_yx_h.set_gl_state(
            preset="translucent",
            blend=True,
            blend_func=("src_alpha", "one_minus_src_alpha"),
            depth_test=False
        )
        # ZY View: V-Line at Z, H-Line at Y
        self.line_zy_v = scene.visuals.InfiniteLine(pos=self.cz, color=(1, 1, 0, 0.5), vertical=True, parent=self.view_zy.scene)
        self.line_zy_h = scene.visuals.InfiniteLine(pos=self.cy, color=(1, 1, 0, 0.5), vertical=False, parent=self.view_zy.scene)
        self.line_zy_v.set_gl_state(
            preset="translucent",
            blend=True,
            blend_func=("src_alpha", "one_minus_src_alpha"),
            depth_test=False
        )
        self.line_zy_h.set_gl_state(
            preset="translucent",
            blend=True,
            blend_func=("src_alpha", "one_minus_src_alpha"),
            depth_test=False
        )
        
        # ZX View: V-Line at X, H-Line at Z
        self.line_zx_v = scene.visuals.InfiniteLine(pos=self.cx, color=(1, 1, 0, 0.5), vertical=True, parent=self.view_zx.scene)
        self.line_zx_h = scene.visuals.InfiniteLine(pos=self.cz, color=(1, 1, 0, 0.5), vertical=False, parent=self.view_zx.scene)
        self.line_zx_v.set_gl_state(
            preset="translucent",
            blend=True,
            blend_func=("src_alpha", "one_minus_src_alpha"),
            depth_test=False
        )
        self.line_zx_h.set_gl_state(
            preset="translucent",
            blend=True,
            blend_func=("src_alpha", "one_minus_src_alpha"),
            depth_test=False
        )
        # -- 4. Controls --
        self.controls_widget = QWidget()
        self.controls_layout = QVBoxLayout(self.controls_widget)
        self.main_layout.addWidget(self.controls_widget)
        
        self._setup_controls()
        self._setup_menu()
        self._setup_events()

        # Initial Update
        self.update_views()
        self.reset_cameras()
        
        # Camera Sync
        # We use Vispy's built-in link functionality.
        # YX (X, Y) <-> ZY (Z, Y) : Sync Y (axis 1)
        self.view_zy.camera.link(self.view_yx.camera, axis='y')
        
        # YX (X, Y) <-> ZX (X, Z) : Sync X (axis 0)
        self.view_zx.camera.link(self.view_yx.camera, axis='x')
        
        # Note: Z-axis (ZY's X and ZX's Y) is left independent for now
        # as cross-axis linking is not directly supported by simple link().

    def _setup_controls(self):
        # Mode (Composite/Single)
        if self.C > 1:
            row = QHBoxLayout()
            row.addWidget(QLabel("Mode:"))
            self.mode_combo = QComboBox()
            self.mode_combo.addItems(["Composite", "Single Channel"])
            self.mode_combo.currentIndexChanged.connect(self.on_mode_change)
            row.addWidget(self.mode_combo)
            row.addStretch()
            self.controls_layout.addLayout(row)
            
            # Channel Slider
            self.channel_row = QWidget()
            l = QHBoxLayout(self.channel_row)
            l.setContentsMargins(0,0,0,0)
            l.addWidget(QLabel("Channel"))
            self.c_slider = QSlider(Qt.Horizontal)
            self.c_slider.setRange(0, self.C - 1)
            self.c_slider.valueChanged.connect(self.on_channel_change)
            l.addWidget(self.c_slider)
            self.controls_layout.addWidget(self.channel_row)
            self.channel_row.setVisible(False)

        # Time Slider
        if self.T > 1:
            row = QHBoxLayout()
            row.addWidget(QLabel("Time"))
            sl = QSlider(Qt.Horizontal)
            sl.setRange(0, self.T - 1)
            sl.valueChanged.connect(self.on_time_change)
            row.addWidget(sl)
            self.controls_layout.addLayout(row)

    def _setup_menu(self):
        menubar = self.menuBar()
        
        # Adjust
        adjust_menu = menubar.addMenu("Adjust")
        bc_action = QAction("Brightness/Contrast", self)
        bc_action.setShortcut("Shift+C")
        bc_action.triggered.connect(self.show_contrast_dialog)
        adjust_menu.addAction(bc_action)
        
        # Image
        image_menu = menubar.addMenu("Image")
        info_action = QAction("Image Info", self)
        info_action.triggered.connect(self.show_metadata_dialog)
        image_menu.addAction(info_action)

    def _setup_events(self):
        # Mouse Events for clicking (add 'Shift' key modifier)
        self.canvas_yx.events.mouse_press.connect(
            lambda e: self.on_shift_click(e, 'yx')
        )
        self.canvas_zy.events.mouse_press.connect(
            lambda e: self.on_shift_click(e, 'zy')
        )
        self.canvas_zx.events.mouse_press.connect(
            lambda e: self.on_shift_click(e, 'zx')
        )


    def on_shift_click(self, e, view_name):

        if e.button != 1:
            return

        # only respond when 'Shift' is held
        if "Shift" not in e.modifiers:
            return
        
        # Map to visual coordinates
        # We use the inverse transform to go from Canvas (Screen) -> Local (Pixel Indices)
        # This handles the STTransform (Scale) automatically.
        if view_name == 'yx':
            tr_to_data = self.vis_yx.layers[0].get_transform(
                map_from="canvas", map_to="visual"
                )
            pos = tr_to_data.map(e.pos)

            x = int(pos[0])
            y = int(pos[1])

            self.cx = np.clip(x, 0, self.X - 1)
            self.cy = np.clip(y, 0, self.Y - 1)
            
        elif view_name == 'zy':
            tr_to_data = self.vis_zy.layers[0].get_transform(
                map_from="canvas", map_to="visual"
                )
            pos = tr_to_data.map(e.pos)
            
            # ZY View Local Coords: (Z, Y) (because we transposed data to (T, X, C, Y, Z) -> sliced X -> (C, Y, Z) -> Vispy sees (Z, Y))
            # Wait, Vispy Image visual sees (col, row) = (x, y).
            # Data is (C, Y, Z).
            # Vispy Image interprets (Y, Z) as (row, col)?
            # Vispy Image(data):
            # If data is (H, W), Vispy displays it with W along X and H along Y.
            # Here data slice is (C, Y, Z) -> (Y, Z) spatial.
            # So Y is height (y-axis), Z is width (x-axis).
            # So local pos[0] is Z, pos[1] is Y.
            
            z = int(pos[0])
            y = int(pos[1])

            self.cz = np.clip(z, 0, self.Z - 1)
            self.cy = np.clip(y, 0, self.Y - 1)
            
        elif view_name == 'zx':
            tr_to_data = self.vis_zx.layers[0].get_transform(
                map_from="canvas", map_to="visual"
                )
            pos = tr_to_data.map(e.pos)
            
            # ZX View Data Slice: (C, Z, X) -> (Z, X) spatial.
            # Z is height (y-axis), X is width (x-axis).
            # So local pos[0] is X, pos[1] is Z.
            
            x = int(pos[0])
            z = int(pos[1])

            self.cx = np.clip(x, 0, self.X - 1)
            self.cz = np.clip(z, 0, self.Z - 1)

        self.update_views()

    def update_views(self):
        # Update Slices
        # YX View: Slice at Z
        self.vis_yx.update_slice(self.ct, self.cz)
        
        # ZY View: Slice at X
        self.vis_zy.update_slice(self.ct, self.cx)
        
        # ZX View: Slice at Y
        self.vis_zx.update_slice(self.ct, self.cy)
        
        # Update Crosshairs (Scaled positions)
        sx, sy, sz = self.scale[2], self.scale[1], self.scale[0]
        
        # YX: V at X, H at Y
        self.line_yx_v.set_data(pos=self.cx * sx)
        self.line_yx_h.set_data(pos=self.cy * sy)
        
        # ZY: V at Z, H at Y
        self.line_zy_v.set_data(pos=self.cz * sz)
        self.line_zy_h.set_data(pos=self.cy * sy)
        
        # ZX: V at X, H at Z
        self.line_zx_v.set_data(pos=self.cx * sx)
        self.line_zx_h.set_data(pos=self.cz * sz)
        
        self.canvas_yx.update()
        self.canvas_zy.update()
        self.canvas_zx.update()

    def reset_cameras(self):
        sx, sy, sz = self.scale[2], self.scale[1], self.scale[0]
        
        # YX
        self.view_yx.camera.rect = (0, 0, self.X * sx, self.Y * sy)
        self.view_yx.camera.flip = (False, True, False)
        
        # ZY
        self.view_zy.camera.rect = (0, 0, self.Z * sz, self.Y * sy)
        self.view_zy.camera.flip = (False, True, False)
        
        # ZX
        self.view_zx.camera.rect = (0, 0, self.X * sx, self.Z * sz)
        self.view_zx.camera.flip = (False, True, False)

    def on_time_change(self, val):
        self.ct = val
        self.update_views()

    def on_mode_change(self, idx):
        mode = "composite" if idx == 0 else "single"
        self.channel_row.setVisible(mode == "single")
        self.proxy.set_mode(mode)
        self.canvas_yx.update()
        self.canvas_zy.update()
        self.canvas_zx.update()

    def on_channel_change(self, val):
        self.proxy.set_active_channel(val)
        self.canvas_yx.update()
        self.canvas_zy.update()
        self.canvas_zx.update()
        
        if hasattr(self, 'contrast_dialog') and self.contrast_dialog.isVisible():
            self.contrast_dialog.combo.setCurrentIndex(val)
            self.contrast_dialog.refresh_ui()

    def show_contrast_dialog(self):
        if not hasattr(self, 'contrast_dialog'):
            # Pass Proxy as viewer
            self.contrast_dialog = ContrastDialog(self, parent=self)
            # Patch the viewer attribute to be our proxy for renderer access
            # But ContrastDialog expects 'viewer.renderer'
            # So we can just set self.renderer = self.proxy temporarily or wrap it?
            # Better: Make ContrastDialog accept an object that has 'renderer' attribute
            # Or just add 'renderer' property to OrthoViewer that returns proxy
            pass
            
        self.contrast_dialog.show()
        self.contrast_dialog.raise_()
        self.contrast_dialog.refresh_ui()
        
    @property
    def renderer(self):
        return self.proxy
        
    @property
    def canvas(self):
        # ContrastDialog calls viewer.canvas.update()
        # We return the YX canvas as a proxy. 
        # Ideally, we should update ALL canvases.
        # We can wrap this in a dummy object or just return one canvas 
        # and ensure our proxy updates the others.
        # Since OrthoVisualProxy updates all visuals, calling update() on one canvas 
        # might not be enough if they are on different canvases?
        # Actually, they ARE on different canvases.
        # So we need a CanvasProxy that updates all 3.
        return self._canvas_proxy

    @property
    def _canvas_proxy(self):
        class CanvasProxy:
            def __init__(self, viewer):
                self.viewer = viewer
            def update(self):
                self.viewer.canvas_yx.update()
                self.viewer.canvas_zy.update()
                self.viewer.canvas_zx.update()
        return CanvasProxy(self)
        


    def show_metadata_dialog(self):
        dlg = MetadataDialog(self.meta, parent=self)
        dlg.exec_()

    def closeEvent(self, event):
        manager.unregister(self)
        super().closeEvent(event)
