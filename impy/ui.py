import sys

import numpy as np
from qtpy import API_NAME
from qtpy.QtCore import Qt
from qtpy.QtGui import QDragEnterEvent, QDropEvent
from qtpy.QtWidgets import (
    QAction,
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSlider,
    QVBoxLayout,
    QWidget,
    QToolBar,
    QButtonGroup,
)
from vispy import app, scene

from .io import load_image
from .visuals import CompositeImageVisual
from .widgets import ContrastDialog, MetadataDialog
from .manager import manager
from .rois import CoordinateROI, RectangleROI, CircleROI, LineROI
from .roi_manager import get_roi_manager

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
        self.canvas = scene.SceneCanvas(
            keys="interactive", bgcolor="black", show=False
        )
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
        self.renderer = CompositeImageVisual(self.view, self.img_data)
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
        self._setup_menu()
        
        # 8. ROI State
        self.rois = []
        self.drawing_roi = None
        self.start_pos = None
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
        super().closeEvent(event)
        
    def focusInEvent(self, event):
        get_roi_manager().set_active_window(self)
        super().focusInEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_A:
            self.renderer.reset_camera(self.img_data.shape)
            self.canvas.update()
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
        
        # Image Menu
        image_menu = menubar.addMenu("Image")
        info_action = QAction("Image Info", self)
        info_action.setShortcut("Shift+I")
        info_action.triggered.connect(self.show_metadata_dialog)
        image_menu.addAction(info_action)

    def show_metadata_dialog(self):
        dlg = MetadataDialog(self.meta, parent=self)
        dlg.exec_()

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
            sl = QSlider(Qt.Horizontal)
            sl.setRange(0, self.Z - 1)
            sl.valueChanged.connect(self.on_z_change)
            row.addWidget(sl)
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

    def on_z_change(self, val):
        self.z_idx = val
        self.update_view()

    def update_view(self):
        self.renderer.update_slice(self.t_idx, self.z_idx)
        self.canvas.update()
        if self.contrast_dialog and self.contrast_dialog.isVisible():
            self.contrast_dialog.refresh_ui()

    def _map_event_to_image(self, event):
        tr = self.canvas.scene.node_transform(self.renderer.layers[0])
        pos = tr.map(event.pos)
        return pos[0], pos[1]

    def on_mouse_press(self, event):
        tool = manager.active_tool
        if tool == "pointer":
            return
            
        x, y = self._map_event_to_image(event)
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
                    self.info_label.setText(f"X: {ix}  Y: {iy}  Val: [{val_str}]")
            else:
                self.info_label.setText("")

        # 2. Update Drawing
        if self.drawing_roi and event.button == 1:
            x, y = self._map_event_to_image(event)
            self.drawing_roi.update(self.start_pos, (x, y))
            self.canvas.update()

    def on_mouse_release(self, event):
        if self.drawing_roi:
            self.drawing_roi = None
            self.start_pos = None


from .roi_manager import get_roi_manager

class Toolbar(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("impy v0.1 (prototype)")
        self.setGeometry(100, 100, 600, 100) # Wider
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
        for f in files:
            self.spawn_viewer(f)

    def open_file_dialog(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open file", ".")
        if fname:
            self.spawn_viewer(fname)

    def closeEvent(self, event):
        # Close all managed windows
        windows = list(manager.get_all().values())
        for w in windows:
            w.close()
            
        # Close ROI Manager
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


def imshow(data, title="Image"):
    """
    Convenience function to show an image from a numpy array.
    """
    # Ensure QApplication exists
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    # Normalize data to 5D (T, Z, C, Y, X)
    # Heuristics for common shapes:
    # 2D: (Y, X) -> (1, 1, 1, Y, X)
    # 3D: (Z, Y, X) -> (1, Z, 1, Y, X) ?? Or (C, Y, X)? Ambiguous.
    # Let's assume (Z, Y, X) for 3D if not specified.
    # 4D: (Z, C, Y, X) -> (1, Z, C, Y, X)
    
    # For simplicity, let's just wrap it if needed, or rely on user to provide correct shape?
    # Better to be robust.
    
    if data.ndim == 2:
        data = data[np.newaxis, np.newaxis, np.newaxis, :, :]
    elif data.ndim == 3:
        # Assume (Z, Y, X)
        data = data[np.newaxis, :, np.newaxis, :, :]
    elif data.ndim == 4:
        # Assume (Z, C, Y, X)
        data = data[np.newaxis, :, :, :, :]
        
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
        app.exec_()
