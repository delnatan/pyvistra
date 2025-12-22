import os
import sys

import numpy as np
from qtpy import API_NAME
from qtpy.QtCore import Qt
from qtpy.QtGui import QDragEnterEvent, QDropEvent
from qtpy.QtWidgets import (
    QAction,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSlider,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
from superqt import QRangeSlider
from vispy import app, scene

from .io import Numpy5DProxy, load_image, normalize_to_5d
from .manager import manager
from .ortho import OrthoViewer
from .roi_manager import get_roi_manager, roi_manager_exists
from .rois import CircleROI, CoordinateROI, LineROI, RectangleROI
from .visuals import CompositeImageVisual
from .widgets import ChannelPanel, ContrastDialog, MetadataDialog

try:
    app.use_app(API_NAME)
except Exception:
    app.use_app("pyqt5")


class ImageWindow(QMainWindow):
    def __init__(self, data_or_path, title="Image"):
        super().__init__()
        self.setAttribute(Qt.WA_DeleteOnClose)

        # 1. Load/Set Data
        if isinstance(data_or_path, str):
            self.filepath = data_or_path
            self.img_data, self.meta = load_image(self.filepath)
            filename = self.meta.get("filename", "Image")
        else:
            self.filepath = None
            # Normalize raw data using helper
            # Note: ImageWindow doesn't take 'dims' arg directly yet,
            # but usually it's called via imshow which does.
            # If instantiated directly, we use default heuristics (dims=None).
            if not isinstance(data_or_path, Numpy5DProxy):
                self.img_data = normalize_to_5d(data_or_path)
            else:
                self.img_data = data_or_path

            self.meta = {}
            filename = title

        # Register with Manager
        self.window_id = manager.register(self)

        self.T, self.Z, self.C, self.Y, self.X = self.img_data.shape

        # Title
        sz, sy, sx = self.meta.get("scale", (1.0, 1.0, 1.0))
        title_str = f"[{self.window_id}] {filename} "
        title_str += f"[{self.X}x{self.Y} px] "
        if self.filepath:
            title_str += f"[{sx:.2f} x {sy:.2f} \u00b5m]"
        self.setWindowTitle(title_str)
        self.resize(700, 750)  # Taller for extra controls

        # 2. Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.layout = QVBoxLayout(central_widget)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        # 3. Vispy Canvas
        self.canvas = scene.SceneCanvas(keys=None, bgcolor="black", show=False)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = "panzoom"
        self.view.camera.aspect = 1

        self.layout.addWidget(self.canvas.native, 1)

        # 4. Info Bar
        self.info_label = QLabel("Hover over image")
        self.info_label.setStyleSheet(
            "background-color: #333; color: #EEE; padding: 4px; font-family: monospace;"
        )
        self.info_label.setFixedHeight(25)
        self.layout.addWidget(self.info_label, 0)

        # 5. Visuals
        is_rgb = self.meta.get("is_rgb", False)
        self.renderer = CompositeImageVisual(self.view, self.img_data, is_rgb=is_rgb)
        self.renderer.reset_camera(self.img_data.shape)

        # 6. Controls Area (Sliders + Mode)
        self.controls_widget = QWidget()
        self.controls_layout = QVBoxLayout(self.controls_widget)
        self.controls_layout.setContentsMargins(10, 10, 10, 10)
        self.controls_layout.setSpacing(5)
        self.layout.addWidget(self.controls_widget, 0)

        self.t_idx = 0
        self.z_idx = 0
        self.c_idx = 0  # Active channel index for Single mode

        self._setup_controls()

        # 7. Menu & Dialogs
        self.contrast_dialog = None
        self.channel_panel = None
        self._setup_menu()

        # 8. ROI State
        self.rois = []
        self.drawing_roi = None
        self.start_pos = None
        # Editing State
        self.dragging_roi = None
        self.drag_handle = None
        self.last_pos = None
        # Toolbar is now external

        # 9. Events
        self.canvas.events.mouse_move.connect(self.on_mouse_move)
        self.canvas.events.mouse_press.connect(self.on_mouse_press)
        self.canvas.events.mouse_release.connect(self.on_mouse_release)

        # Focus policy
        self.setFocusPolicy(Qt.StrongFocus)

        # Initial Draw
        self.update_view()

    def closeEvent(self, event):
        manager.unregister(self)
        get_roi_manager().remove_window(self)
        super().closeEvent(event)

    def focusInEvent(self, event):
        get_roi_manager().set_active_window(self)
        super().focusInEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_A:
            self.renderer.reset_camera(self.img_data.shape)
            self.canvas.update()
        elif event.key() == Qt.Key_F:
            # Flip selected CoordinateROI
            for roi in self.rois:
                if roi.selected and isinstance(roi, CoordinateROI):
                    roi.flip()
                    self.canvas.update()
                    break
        elif event.key() == Qt.Key_L:
            # Toggle ROI labels visibility
            from .rois import ROI
            show = ROI.toggle_labels()
            # Update visibility for all ROIs in all windows
            for w in manager.get_all().values():
                for roi in w.rois:
                    roi.label_visual.visible = show
                w.canvas.update()
        elif event.key() == Qt.Key_Escape:
            # Deselect all ROIs
            for roi in self.rois:
                roi.select(False)
            self.canvas.update()
            # Notify Manager (optional, but good for sync)
            get_roi_manager().select_roi(None)
        else:
            super().keyPressEvent(event)

    def get_data(self):
        """Return the current image data."""
        return self.img_data

    def set_data(self, new_data):
        """Update the image data in place."""
        if new_data.ndim != 5:
            # Try to reshape or warn? For now assume 5D or compatible
            pass

        self.img_data = new_data
        # Update renderer data
        self.renderer.data = new_data
        self.renderer.update_slice(self.t_idx, self.z_idx)
        self.canvas.update()

    def _setup_menu(self):
        menubar = self.menuBar()

        # Adjust Menu
        adjust_menu = menubar.addMenu("Adjust")
        bc_action = QAction("Brightness/Contrast", self)
        bc_action.setShortcut("Shift+C")
        bc_action.triggered.connect(self.show_contrast_dialog)
        adjust_menu.addAction(bc_action)

        channels_action = QAction("Channels...", self)
        channels_action.setShortcut("Shift+H")
        channels_action.triggered.connect(self.show_channel_panel)
        adjust_menu.addAction(channels_action)

        # Image Menu
        image_menu = menubar.addMenu("Image")
        info_action = QAction("Image Info", self)
        info_action.setShortcut("Shift+I")
        info_action.triggered.connect(self.show_metadata_dialog)
        image_menu.addAction(info_action)

        ortho_action = QAction("Ortho View", self)
        ortho_action.triggered.connect(self.show_ortho_view)
        image_menu.addAction(ortho_action)

    def show_metadata_dialog(self):
        dlg = MetadataDialog(self.meta, parent=self)
        dlg.exec_()

    def show_ortho_view(self):
        self.ortho_viewer = OrthoViewer(
            self.img_data,
            self.meta,
            title=f"Ortho View - {self.windowTitle()}",
        )
        self.ortho_viewer.show()

    def update_cursor(self):
        tool = manager.active_tool
        if tool == "pointer":
            self.view.camera.interactive = True
        else:
            self.view.camera.interactive = False

    def show_contrast_dialog(self):
        if self.contrast_dialog is None:
            self.contrast_dialog = ContrastDialog(self, parent=self)
        self.contrast_dialog.show()
        self.contrast_dialog.raise_()
        self.contrast_dialog.refresh_ui()

    def show_channel_panel(self):
        if self.channel_panel is None:
            self.channel_panel = ChannelPanel(self, parent=self)
        self.channel_panel.show()
        self.channel_panel.raise_()
        self.channel_panel.refresh_ui()

    def set_tool(self, tool_name):
        """
        Set the active tool (e.g. 'pointer', 'rect', 'circle', 'line', 'coordinate').
        """
        valid_tools = ["pointer", "coordinate", "rect", "circle", "line"]
        if tool_name not in valid_tools:
            print(f"Invalid tool: {tool_name}. Valid tools: {valid_tools}")
            return

        manager.active_tool = tool_name

        # Update cursors in all windows
        for w in manager.get_all().values():
            w.update_cursor()

    def _setup_controls(self):
        # -- Mode Selector (Only if Multi-channel) --
        if self.C > 1:
            row = QHBoxLayout()
            row.addWidget(QLabel("Mode:"))

            self.mode_combo = QComboBox()
            self.mode_combo.addItems(["Composite", "Single Channel"])
            self.mode_combo.currentIndexChanged.connect(self.on_mode_change)
            row.addWidget(self.mode_combo)
            row.addStretch()
            self.controls_layout.addLayout(row)

            # -- Channel Slider (Initially Hidden) --
            self.channel_row_widget = QWidget()
            c_layout = QHBoxLayout(self.channel_row_widget)
            c_layout.setContentsMargins(0, 0, 0, 0)

            c_layout.addWidget(QLabel("Channel"))
            self.c_slider = QSlider(Qt.Horizontal)
            self.c_slider.setRange(0, self.C - 1)
            self.c_slider.valueChanged.connect(self.on_channel_change)
            c_layout.addWidget(self.c_slider)

            self.controls_layout.addWidget(self.channel_row_widget)
            self.channel_row_widget.setVisible(False)  # Default is Composite

        # -- Time Slider --
        if self.T > 1:
            row = QHBoxLayout()
            row.addWidget(QLabel("Time"))
            sl = QSlider(Qt.Horizontal)
            sl.setRange(0, self.T - 1)
            sl.valueChanged.connect(self.on_time_change)
            row.addWidget(sl)
            self.controls_layout.addLayout(row)

        # -- Z Slider --
        if self.Z > 1:
            row = QHBoxLayout()
            row.addWidget(QLabel("Z-Pos"))

            # Standard Slider
            self.z_slider = QSlider(Qt.Horizontal)
            self.z_slider.setRange(0, self.Z - 1)
            self.z_slider.valueChanged.connect(self.on_z_change)
            row.addWidget(self.z_slider)

            self.z_label = QLabel("0")
            self.z_label.setFixedWidth(30)  # Fixed width to prevent jumping
            self.z_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row.addWidget(self.z_label)

            # Projection Controls
            self.chk_proj = QCheckBox("Max Proj")
            self.chk_proj.toggled.connect(self.toggle_z_projection)
            self.z_range_slider_widget = QWidget()
            self.z_range_slider_layout = QHBoxLayout()
            self.z_range_slider_layout.setContentsMargins(0, 0, 0, 0)
            self.z_range_slider = QRangeSlider(Qt.Horizontal)
            self.z_range_slider_min_label = QLabel("0")
            self.z_range_slider_max_label = QLabel(f"{self.Z - 1}")
            self.z_range_slider.setRange(0, self.Z - 1)
            self.z_range_slider.setValue((0, self.Z - 1))
            self.z_range_slider.barIsVisible = True
            self.z_range_slider.barIsEnabled = True
            self.z_range_slider.barIsEnabled = False

            self.z_range_slider_layout.addWidget(self.z_range_slider_min_label)
            self.z_range_slider_layout.addWidget(self.z_range_slider)
            self.z_range_slider_layout.addWidget(self.z_range_slider_max_label)
            self.z_range_slider_widget.setLayout(self.z_range_slider_layout)
            self.z_range_slider_widget.setVisible(False)

            self.z_range_slider.valueChanged.connect(self.on_z_proj_change)
            row.addWidget(self.z_range_slider_widget)
            row.addWidget(self.chk_proj)
            self.controls_layout.addLayout(row)

    def on_mode_change(self, index):
        mode = "composite" if index == 0 else "single"

        # Toggle Channel Slider Visibility
        if self.C > 1:
            self.channel_row_widget.setVisible(mode == "single")

        # Update Renderer
        self.renderer.set_mode(mode)
        self.canvas.update()

    def on_channel_change(self, val):
        self.c_idx = val
        self.renderer.set_active_channel(val)
        self.canvas.update()

        # If Contrast Dialog is open, sync it to this channel
        if self.contrast_dialog and self.contrast_dialog.isVisible():
            self.contrast_dialog.combo.setCurrentIndex(val)
            self.contrast_dialog.refresh_ui()

    def on_time_change(self, val):
        self.t_idx = val
        self.update_view()

    def toggle_z_projection(self, checked):
        self.z_slider.setVisible(not checked)
        self.z_range_slider_widget.setVisible(checked)
        self.update_view()

    def on_z_proj_change(self, val):
        # update z-min/max labels
        self.z_range_slider_min_label.setText(str(val[0]))
        self.z_range_slider_max_label.setText(str(val[1]))
        self.update_view()

    def on_z_change(self, val):
        self.z_idx = val
        if hasattr(self, "z_label"):
            self.z_label.setText(str(val))
        self.update_view()

    def update_view(self):
        if hasattr(self, "chk_proj") and self.chk_proj.isChecked():
            mn, mx = self.z_range_slider.value()
            z_slice = slice(mn, mx + 1)
            self.renderer.update_slice(self.t_idx, z_slice)
        else:
            self.renderer.update_slice(self.t_idx, self.z_idx)

        self.canvas.update()
        if self.contrast_dialog and self.contrast_dialog.isVisible():
            self.contrast_dialog.refresh_ui()
        if self.channel_panel and self.channel_panel.isVisible():
            self.channel_panel.refresh_ui()

    def _map_event_to_image(self, event):
        tr = self.canvas.scene.node_transform(self.renderer.layers[0])
        pos = tr.map(event.pos)
        return pos[0], pos[1]

    def on_mouse_press(self, event):
        tool = manager.active_tool
        x, y = self._map_event_to_image(event)

        if tool == "pointer":
            # Hit Test (Reverse order to select top-most)
            hit_roi = None
            hit_handle = None

            for roi in reversed(self.rois):
                res = roi.hit_test((x, y))
                if res:
                    hit_roi = roi
                    hit_handle = res
                    break

            # Update Selection
            for roi in self.rois:
                roi.select(roi is hit_roi)

            # Notify Manager
            get_roi_manager().select_roi(hit_roi)

            if hit_roi:
                self.dragging_roi = hit_roi
                self.drag_handle = hit_handle
                self.last_pos = (x, y)
                self.canvas.update()
            else:
                self.canvas.update()
            return

        self.start_pos = (x, y)

        if tool == "coordinate":
            self.drawing_roi = CoordinateROI(self.view)
        elif tool == "rect":
            self.drawing_roi = RectangleROI(self.view)
        elif tool == "circle":
            self.drawing_roi = CircleROI(self.view)
        elif tool == "line":
            self.drawing_roi = LineROI(self.view)

        if self.drawing_roi:
            self.rois.append(self.drawing_roi)
            get_roi_manager().add_roi(self.drawing_roi)
            # Initial update (zero size/length)
            self.drawing_roi.update((x, y), (x, y))
            self.canvas.update()

    def on_mouse_move(self, event):
        # 1. Update Info Label (always)
        if self.renderer.layers:
            x, y = self._map_event_to_image(event)
            ix, iy = int(x), int(y)
            if 0 <= ix < self.X and 0 <= iy < self.Y:
                cache = self.renderer.current_slice_cache
                if cache is not None:
                    vals = []
                    for c in range(cache.shape[0]):
                        try:
                            val = cache[c, iy, ix]
                            vals.append(f"{val:.1f}")
                        except IndexError:
                            pass
                    val_str = ", ".join(vals)
                    self.info_label.setText(
                        f"X: {ix}  Y: {iy}  Val: [{val_str}]"
                    )
            else:
                self.info_label.setText("")

        # 2. ROI Editing
        if self.dragging_roi and event.button == 1:
            x, y = self._map_event_to_image(event)
            dx = x - self.last_pos[0]
            dy = y - self.last_pos[1]

            if self.drag_handle == "center":
                self.dragging_roi.move((dx, dy))
            else:
                self.dragging_roi.adjust(self.drag_handle, (x, y))

            self.last_pos = (x, y)
            self.canvas.update()
            return

        # 3. Update Drawing
        if self.drawing_roi and event.button == 1:
            x, y = self._map_event_to_image(event)
            end_pos = (x, y)

            # Shift key constrains LineROI to horizontal/vertical
            if isinstance(self.drawing_roi, LineROI) and 'Shift' in event.modifiers:
                sx, sy = self.start_pos
                dx = abs(x - sx)
                dy = abs(y - sy)
                if dx > dy:
                    # Horizontal line
                    end_pos = (x, sy)
                else:
                    # Vertical line
                    end_pos = (sx, y)

            self.drawing_roi.update(self.start_pos, end_pos)
            self.canvas.update()

    def on_mouse_release(self, event):
        if self.dragging_roi:
            self.dragging_roi = None
            self.drag_handle = None
            self.last_pos = None

        if self.drawing_roi:
            self.drawing_roi = None
            self.start_pos = None


class Toolbar(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("pyvistra v0.1 (prototype)")
        self.setGeometry(100, 100, 600, 100)  # Wider
        self.setAcceptDrops(True)
        self.open_windows = []

        # Central Widget with Layout
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Label
        self.label = QLabel("Drag & Drop Images")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        # Tool Bar (Actual QToolBar)
        self.tools = QToolBar("Tools")
        self.addToolBar(self.tools)

        # Actions
        self.act_pointer = QAction("Pointer", self)
        self.act_pointer.setCheckable(True)
        self.act_pointer.setChecked(True)
        self.act_pointer.triggered.connect(lambda: self.set_tool("pointer"))

        self.act_coord = QAction("Coordinate", self)
        self.act_coord.setCheckable(True)
        self.act_coord.triggered.connect(lambda: self.set_tool("coordinate"))

        self.act_rect = QAction("Rectangle", self)
        self.act_rect.setCheckable(True)
        self.act_rect.triggered.connect(lambda: self.set_tool("rect"))

        self.act_circle = QAction("Circle", self)
        self.act_circle.setCheckable(True)
        self.act_circle.triggered.connect(lambda: self.set_tool("circle"))

        self.act_line = QAction("Line", self)
        self.act_line.setCheckable(True)
        self.act_line.triggered.connect(lambda: self.set_tool("line"))

        self.tools.addAction(self.act_pointer)
        self.tools.addAction(self.act_coord)
        self.tools.addAction(self.act_rect)
        self.tools.addAction(self.act_circle)
        self.tools.addAction(self.act_line)

        # ROI Manager Button
        self.tools.addSeparator()
        self.act_roi_mgr = QAction("ROI Manager", self)
        self.act_roi_mgr.triggered.connect(self.show_roi_manager)
        self.tools.addAction(self.act_roi_mgr)

        # Group
        from qtpy.QtWidgets import QActionGroup

        group = QActionGroup(self)
        group.addAction(self.act_pointer)
        group.addAction(self.act_coord)
        group.addAction(self.act_rect)
        group.addAction(self.act_circle)
        group.addAction(self.act_line)

        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        open_action = QAction("Open", self)
        open_action.triggered.connect(self.open_file_dialog)
        file_menu.addAction(open_action)

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def set_tool(self, tool_name):
        manager.active_tool = tool_name
        # Update cursor or state in all windows?
        # For now, windows check state on click.
        # But we might want to update cursor immediately.
        # Let's iterate windows
        for w in manager.get_all().values():
            w.update_cursor()

    def show_roi_manager(self):
        mgr = get_roi_manager()
        mgr.show()
        mgr.raise_()

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        files = [u.toLocalFile() for u in event.mimeData().urls()]

        # Collect supported image files
        supported_ext = {".ims", ".tif", ".tiff", ".png", ".jpg", ".jpeg"}
        image_files = []

        for f in files:
            if os.path.isdir(f):
                # Folder: collect all images recursively
                for root, _, names in os.walk(f):
                    for name in names:
                        if os.path.splitext(name)[1].lower() in supported_ext:
                            image_files.append(os.path.join(root, name))
            elif os.path.splitext(f)[1].lower() in supported_ext:
                image_files.append(f)

        # Sort by filename
        image_files.sort()

        if len(image_files) > 1:
            # Multiple files -> TiledViewer
            from .tiled_viewer import TiledViewer

            viewer = TiledViewer(image_files)
            viewer.show()
            self.open_windows.append(viewer)
        elif len(image_files) == 1:
            # Single file -> regular ImageWindow
            self.spawn_viewer(image_files[0])

    def open_file_dialog(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open file", ".")
        if fname:
            self.spawn_viewer(fname)

    def closeEvent(self, event):
        # Close all managed windows
        windows = list(manager.get_all().values())
        for w in windows:
            w.close()

        # Close ROI Manager only if it was already created
        # Avoids creating a widget during shutdown which causes segfault
        if roi_manager_exists():
            try:
                mgr = get_roi_manager()
                if mgr.isVisible():
                    mgr.close()
            except Exception:
                pass

        super().closeEvent(event)

    def spawn_viewer(self, filepath):
        try:
            viewer = ImageWindow(filepath)
            viewer.show()
            self.open_windows.append(viewer)
        except Exception as e:
            print(f"Error opening {filepath}: {e}")


def imshow(data, title="Image", dims=None):
    """
    Convenience function to show an image from a numpy array.

    Args:
        data (np.ndarray): Image data.
        title (str): Window title.
        dims (str): Dimension order string (e.g. 'tyx', 'zcyx').
                    If None, heuristics are used.
    """
    # Ensure QApplication exists
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # Apply Theme
    from .theme import DARK_THEME

    app.setStyleSheet(DARK_THEME)

    # Normalize data to 5D (T, Z, C, Y, X)
    data = normalize_to_5d(data, dims=dims)

    viewer = ImageWindow(data, title=title)
    viewer.show()

    # If running in interactive shell, we might not want to block?
    # But usually we need app.exec_() if not in IPython/Jupyter with event loop integration.
    # For now, just return the viewer.
    return viewer


def run_app():
    """
    Start the Qt event loop. Use this when running from a script
    to ensure windows are visible and interactive.
    """
    app = QApplication.instance()
    if app:
        from .theme import DARK_THEME

        app.setStyleSheet(DARK_THEME)
        app.exec_()
