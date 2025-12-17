"""
Tiled Image Viewer for displaying multiple images in a gallery layout.

Inspired by single-particle cryo-EM software (RELION, cryoSPARC).
Supports paginated viewing of many images with GPU-accelerated rendering.
"""

import os
from pathlib import Path

import numpy as np
from qtpy.QtCore import Qt, QRect, QPoint, QSize
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLayoutItem,
    QMainWindow,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QWidgetItem,
)
from vispy import scene

from .io import load_image
from .visuals import CompositeImageVisual
from .widgets import ContrastDialog


class FlowLayout(QLayout):
    """
    Layout that arranges widgets in a flowing left-to-right, top-to-bottom manner.
    Similar to CSS flexbox with wrap.
    """

    def __init__(self, parent=None, spacing=8):
        super().__init__(parent)
        self._items = []
        self._spacing = spacing

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self):
        return Qt.Orientations()

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margin = self.contentsMargins()
        size += QSize(margin.left() + margin.right(), margin.top() + margin.bottom())
        return size

    def _do_layout(self, rect, test_only=False):
        """Arrange items in flow layout within given rect."""
        left, top, right, bottom = self.getContentsMargins()
        effective_rect = rect.adjusted(left, top, -right, -bottom)

        x = effective_rect.x()
        y = effective_rect.y()
        line_height = 0

        for item in self._items:
            widget = item.widget()
            if widget is None:
                continue

            space_x = self._spacing
            space_y = self._spacing

            item_width = widget.sizeHint().width()
            item_height = widget.sizeHint().height()

            next_x = x + item_width + space_x

            # Wrap to next row if exceeded width
            if next_x - space_x > effective_rect.right() + 1 and line_height > 0:
                x = effective_rect.x()
                y = y + line_height + space_y
                next_x = x + item_width + space_x
                line_height = 0

            if not test_only:
                widget.setGeometry(QRect(QPoint(x, y), QSize(item_width, item_height)))

            x = next_x
            line_height = max(line_height, item_height)

        return y + line_height - rect.y() + bottom


class TileWidget(QFrame):
    """
    Single tile displaying one image with Vispy canvas.
    Supports independent pan/zoom within the tile.
    """

    def __init__(self, tile_size=200, parent=None):
        super().__init__(parent)
        self._tile_size = tile_size
        self._parent_viewer = parent

        # Data
        self.data = None
        self.meta = None
        self.file_path = None

        # Setup UI
        self._setup_ui()

        # Context menu
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

    def _setup_ui(self):
        self.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.setLineWidth(1)
        self.setStyleSheet(
            """
            TileWidget {
                background-color: #1a1a1a;
                border: 1px solid #333;
            }
            TileWidget:hover {
                border: 1px solid #555;
            }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        # Vispy canvas
        self.canvas = scene.SceneCanvas(keys=None, bgcolor="#1a1a1a", show=False)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = "panzoom"
        self.view.camera.aspect = 1

        # Canvas widget
        self.canvas.native.setMinimumSize(50, 50)
        layout.addWidget(self.canvas.native, 1)

        # Info label
        self.info_label = QLabel("")
        self.info_label.setStyleSheet(
            "color: #aaa; font-size: 10px; padding: 2px; background: transparent;"
        )
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setWordWrap(True)
        self.info_label.setFixedHeight(18)
        layout.addWidget(self.info_label, 0)

        # Renderer (created when data is loaded)
        self.renderer = None

        self._update_size()

    def _update_size(self):
        """Update widget size based on tile_size."""
        label_height = 18
        total_height = self._tile_size + label_height + 8  # margins + spacing
        self.setFixedSize(self._tile_size, total_height)

    def sizeHint(self):
        label_height = 18
        total_height = self._tile_size + label_height + 8
        return QSize(self._tile_size, total_height)

    def set_tile_size(self, size):
        """Update tile size (does not affect internal zoom)."""
        self._tile_size = size
        self._update_size()
        if self.renderer is not None:
            self._fit_view()

    def load(self, path):
        """Load image from path."""
        self.file_path = path
        try:
            self.data, self.meta = load_image(path)

            # Create renderer
            self.renderer = CompositeImageVisual(self.view, self.data)

            # Display first frame/slice (middle Z)
            z_mid = self.data.shape[1] // 2
            self.renderer.update_slice(0, z_mid)
            self._fit_view()
            self._update_info()

        except Exception as e:
            print(f"Error loading {path}: {e}")
            self.info_label.setText("Load error")
            self.info_label.setStyleSheet("color: #ff6666; font-size: 10px;")

    def unload(self):
        """Release memory."""
        if self.renderer is not None:
            # Clear visuals
            for layer in self.renderer.layers:
                layer.parent = None
            self.renderer = None
        self.data = None
        self.meta = None

    def _fit_view(self):
        """Reset camera to fit image."""
        if self.data is not None:
            _, _, _, h, w = self.data.shape
            self.view.camera.set_range(x=(0, w), y=(0, h), margin=0.02)
            self.view.camera.flip = (False, True, False)

    def _update_info(self):
        """Update info label with filename."""
        if self.file_path:
            name = Path(self.file_path).stem
            # Truncate if too long
            max_len = self._tile_size // 8
            if len(name) > max_len:
                name = name[: max_len - 2] + ".."
            self.info_label.setText(name)
            self.info_label.setToolTip(self.file_path)

    def _show_context_menu(self, pos):
        """Show right-click context menu."""
        menu = QMenu(self)

        # Auto contrast
        auto_action = menu.addAction("Auto Contrast")
        auto_action.triggered.connect(self._auto_contrast)

        # Contrast dialog
        contrast_action = menu.addAction("Adjust Contrast...")
        contrast_action.triggered.connect(self._show_contrast_dialog)

        menu.addSeparator()

        # Reset view
        reset_action = menu.addAction("Reset View")
        reset_action.triggered.connect(self._fit_view)

        menu.addSeparator()

        # Open in viewer
        open_action = menu.addAction("Open in Viewer")
        open_action.triggered.connect(self._open_in_viewer)

        # Show metadata
        meta_action = menu.addAction("Show Info")
        meta_action.triggered.connect(self._show_metadata)

        menu.exec_(self.mapToGlobal(pos))

    def _auto_contrast(self):
        """Apply auto-contrast to all channels."""
        if self.renderer is None or self.renderer.current_slice_cache is None:
            return

        cache = self.renderer.current_slice_cache
        for c in range(cache.shape[0]):
            plane = cache[c]
            valid = plane[plane > 0]
            if valid.size > 0:
                mn, mx = np.nanpercentile(valid, (0.5, 99.5))
                self.renderer.set_clim(c, mn, max(mx, mn + 1))

        self.canvas.update()

    def _show_contrast_dialog(self):
        """Show contrast adjustment dialog."""
        if self.renderer is None:
            return

        # Create a temporary ImageWindow-like wrapper for ContrastDialog
        class TileWrapper:
            def __init__(self, tile):
                self.renderer = tile.renderer
                self.canvas = tile.canvas
                self.C = tile.data.shape[2] if tile.data is not None else 1
                self.meta = tile.meta or {}

        wrapper = TileWrapper(self)
        dlg = ContrastDialog(wrapper, parent=self)
        dlg.exec_()

    def _open_in_viewer(self):
        """Open this image in a full ImageWindow."""
        if self.file_path:
            from .ui import ImageWindow

            viewer = ImageWindow(self.file_path)
            viewer.show()

    def _show_metadata(self):
        """Show metadata dialog."""
        if self.meta:
            from .widgets import MetadataDialog

            dlg = MetadataDialog(self.meta, parent=self)
            dlg.exec_()


class TiledViewer(QMainWindow):
    """
    Gallery view for multiple images with flow layout and pagination.
    """

    # Maximum tiles per page options
    TILES_PER_PAGE_OPTIONS = [25, 50, 100]

    def __init__(self, image_paths, tiles_per_page=50, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose)

        self.image_paths = sorted(image_paths)
        self.tiles_per_page = tiles_per_page
        self.current_page = 0
        self.tile_size = 200  # Default tile size in pixels

        self.tile_widgets = []

        self._setup_ui()
        self._load_current_page()

    def _setup_ui(self):
        self.setWindowTitle(f"Tiled Viewer - {len(self.image_paths)} images")
        self.resize(1000, 800)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # Toolbar area
        toolbar = QWidget()
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(5, 5, 5, 5)

        # Page navigation
        self.prev_btn = QPushButton("<")
        self.prev_btn.setFixedWidth(30)
        self.prev_btn.clicked.connect(self._prev_page)
        toolbar_layout.addWidget(self.prev_btn)

        self.page_label = QLabel("Page 1 / 1")
        self.page_label.setAlignment(Qt.AlignCenter)
        self.page_label.setFixedWidth(100)
        toolbar_layout.addWidget(self.page_label)

        self.next_btn = QPushButton(">")
        self.next_btn.setFixedWidth(30)
        self.next_btn.clicked.connect(self._next_page)
        toolbar_layout.addWidget(self.next_btn)

        toolbar_layout.addSpacing(20)

        # Tiles per page
        toolbar_layout.addWidget(QLabel("Tiles/page:"))
        self.per_page_combo = QComboBox()
        for n in self.TILES_PER_PAGE_OPTIONS:
            self.per_page_combo.addItem(str(n), n)
        # Set current value
        idx = (
            self.TILES_PER_PAGE_OPTIONS.index(self.tiles_per_page)
            if self.tiles_per_page in self.TILES_PER_PAGE_OPTIONS
            else 1
        )
        self.per_page_combo.setCurrentIndex(idx)
        self.per_page_combo.currentIndexChanged.connect(self._on_per_page_changed)
        toolbar_layout.addWidget(self.per_page_combo)

        toolbar_layout.addSpacing(20)

        # Tile size slider
        toolbar_layout.addWidget(QLabel("Tile size:"))
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setRange(100, 400)
        self.size_slider.setValue(self.tile_size)
        self.size_slider.setFixedWidth(150)
        self.size_slider.valueChanged.connect(self._on_tile_size_changed)
        toolbar_layout.addWidget(self.size_slider)

        self.size_label = QLabel(f"{self.tile_size}px")
        self.size_label.setFixedWidth(50)
        toolbar_layout.addWidget(self.size_label)

        toolbar_layout.addSpacing(20)

        # Auto contrast all button
        self.auto_all_btn = QPushButton("Auto All")
        self.auto_all_btn.clicked.connect(self._auto_contrast_all)
        toolbar_layout.addWidget(self.auto_all_btn)

        toolbar_layout.addStretch()

        main_layout.addWidget(toolbar)

        # Scroll area for tiles
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Container with flow layout
        self.flow_container = QWidget()
        self.flow_layout = FlowLayout(self.flow_container, spacing=8)
        self.scroll_area.setWidget(self.flow_container)

        main_layout.addWidget(self.scroll_area, 1)

        # Status bar
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #888; padding: 5px;")
        main_layout.addWidget(self.status_label)

        self._update_page_controls()

    def _total_pages(self):
        return max(1, (len(self.image_paths) + self.tiles_per_page - 1) // self.tiles_per_page)

    def _update_page_controls(self):
        """Update page navigation UI."""
        total = self._total_pages()
        self.page_label.setText(f"Page {self.current_page + 1} / {total}")
        self.prev_btn.setEnabled(self.current_page > 0)
        self.next_btn.setEnabled(self.current_page < total - 1)

    def _update_status(self):
        """Update status bar."""
        start = self.current_page * self.tiles_per_page + 1
        end = min(start + len(self.tile_widgets) - 1, len(self.image_paths))
        self.status_label.setText(
            f"Showing {start}-{end} of {len(self.image_paths)} images"
        )

    def _load_current_page(self):
        """Load tiles for current page."""
        # Clear existing tiles
        self._clear_tiles()

        # Calculate slice
        start = self.current_page * self.tiles_per_page
        end = min(start + self.tiles_per_page, len(self.image_paths))

        # Create and load tiles
        for path in self.image_paths[start:end]:
            tile = TileWidget(self.tile_size, parent=self)
            tile.load(path)
            self.flow_layout.addWidget(tile)
            self.tile_widgets.append(tile)

        # Update UI
        self._update_page_controls()
        self._update_status()

        # Force layout update
        self.flow_container.adjustSize()

    def _clear_tiles(self):
        """Remove and unload all current tiles."""
        for tile in self.tile_widgets:
            tile.unload()
            self.flow_layout.removeWidget(tile)
            tile.deleteLater()
        self.tile_widgets.clear()

    def _prev_page(self):
        """Go to previous page."""
        if self.current_page > 0:
            self.current_page -= 1
            self._load_current_page()

    def _next_page(self):
        """Go to next page."""
        if self.current_page < self._total_pages() - 1:
            self.current_page += 1
            self._load_current_page()

    def _on_per_page_changed(self, index):
        """Handle tiles per page change."""
        self.tiles_per_page = self.per_page_combo.currentData()
        # Reset to first page and reload
        self.current_page = 0
        self._load_current_page()

    def _on_tile_size_changed(self, value):
        """Handle tile size slider change."""
        self.tile_size = value
        self.size_label.setText(f"{value}px")

        for tile in self.tile_widgets:
            tile.set_tile_size(value)

        # Trigger reflow
        self.flow_container.adjustSize()

    def _auto_contrast_all(self):
        """Apply auto-contrast to all visible tiles."""
        for tile in self.tile_widgets:
            tile._auto_contrast()

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts."""
        key = event.key()

        if key == Qt.Key_Left or key == Qt.Key_PageUp:
            self._prev_page()
        elif key == Qt.Key_Right or key == Qt.Key_PageDown:
            self._next_page()
        elif key == Qt.Key_Home:
            self.current_page = 0
            self._load_current_page()
        elif key == Qt.Key_End:
            self.current_page = self._total_pages() - 1
            self._load_current_page()
        elif key == Qt.Key_A:
            self._auto_contrast_all()
        elif key == Qt.Key_Plus or key == Qt.Key_Equal:
            # Increase tile size
            new_size = min(400, self.tile_size + 25)
            self.size_slider.setValue(new_size)
        elif key == Qt.Key_Minus:
            # Decrease tile size
            new_size = max(100, self.tile_size - 25)
            self.size_slider.setValue(new_size)
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        """Clean up on close."""
        self._clear_tiles()
        super().closeEvent(event)


def open_tiled_viewer(image_paths, tiles_per_page=50):
    """
    Convenience function to open a tiled viewer.

    Args:
        image_paths: List of image file paths
        tiles_per_page: Maximum tiles per page (default 50)

    Returns:
        TiledViewer instance
    """
    # Ensure QApplication exists
    app = QApplication.instance()
    if app is None:
        import sys
        app = QApplication(sys.argv)

    # Apply theme
    from .theme import DARK_THEME
    app.setStyleSheet(DARK_THEME)

    viewer = TiledViewer(image_paths, tiles_per_page=tiles_per_page)
    viewer.show()

    return viewer
