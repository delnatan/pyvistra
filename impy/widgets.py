import numpy as np
from qtpy.QtCore import Qt, Signal, QRectF
from qtpy.QtGui import QBrush, QColor, QFont, QPainter, QPen
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QSlider,
    QDoubleSpinBox,
)

# Theme Constants
WIDGET_BG = QColor(32, 32, 32)
TEXT_COLOR = QColor(224, 224, 224)
HANDLE_COLOR = QColor(255, 255, 255)
HANDLE_WIDTH = 6


class HistogramWidget(QWidget):
    """
    Interactive Histogram Widget.
    Displays a log-histogram and allows dragging two handles (min/max)
    to adjust contrast limits.
    """

    climChanged = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(120)
        self.setMouseTracking(True)

        # Data
        self.hist_data = None
        self.data_min = 0.0
        self.data_max = 1.0
        self.color = TEXT_COLOR

        # State
        self.clim_min = 0.0
        self.clim_max = 1.0
        
        # Interaction
        self._dragging = None  # 'min', 'max', 'center', or None
        self._last_mouse_x = 0

    def set_data(self, data_slice, color_name):
        # 1. Compute Histogram
        # We use a fixed number of bins for display
        self.data_min = float(np.nanmin(data_slice))
        self.data_max = float(np.nanmax(data_slice))
        
        if self.data_max <= self.data_min:
            self.data_max = self.data_min + 1e-5

        y, x = np.histogram(data_slice, bins=100, range=(self.data_min, self.data_max))
        self.hist_data = np.log1p(y)
        
        self.color = QColor(color_name)
        self.update()

    def set_clim(self, vmin, vmax):
        self.clim_min = vmin
        self.clim_max = vmax
        self.update()

    def _val_to_x(self, val):
        w = self.width()
        span = self.data_max - self.data_min
        if span <= 0: return 0
        ratio = (val - self.data_min) / span
        return int(ratio * w)

    def _x_to_val(self, x):
        w = self.width()
        span = self.data_max - self.data_min
        ratio = x / w
        val = self.data_min + (ratio * span)
        return max(self.data_min, min(val, self.data_max))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), WIDGET_BG)

        w = self.width()
        h = self.height()

        # 1. Draw Histogram
        if self.hist_data is not None:
            max_log = np.max(self.hist_data)
            if max_log == 0: max_log = 1
            
            fill_color = QColor(self.color)
            fill_color.setAlpha(100)
            painter.setBrush(QBrush(fill_color))
            painter.setPen(Qt.NoPen)

            n_bins = len(self.hist_data)
            bin_w = w / n_bins

            for i, val in enumerate(self.hist_data):
                bar_h = (val / max_log) * (h - 20)
                x = i * bin_w
                y = h - bar_h
                painter.drawRect(QRectF(x, y, bin_w, bar_h))

        # 2. Draw Overlay (Darken outside selection)
        x_min = self._val_to_x(self.clim_min)
        x_max = self._val_to_x(self.clim_max)

        dark_overlay = QColor(0, 0, 0, 150)
        painter.fillRect(0, 0, x_min, h, dark_overlay)
        painter.fillRect(x_max, 0, w - x_max, h, dark_overlay)

        # 3. Draw Handles
        pen = QPen(HANDLE_COLOR)
        pen.setWidth(2)
        painter.setPen(pen)
        
        # Min Handle
        painter.drawLine(x_min, 0, x_min, h)
        # Max Handle
        painter.drawLine(x_max, 0, x_max, h)

        # 4. Text Labels
        painter.setPen(TEXT_COLOR)
        font = QFont("Segoe UI", 8)
        painter.setFont(font)

        # Draw min/max values at handles
        min_str = f"{self.clim_min:.1f}"
        max_str = f"{self.clim_max:.1f}"
        
        # Adjust text position to stay on screen
        fm = painter.fontMetrics()
        tw_min = fm.width(min_str)
        tw_max = fm.width(max_str)

        draw_x_min = max(2, min(x_min - tw_min - 2, w - tw_min - 2))
        draw_x_max = min(w - tw_max - 2, max(x_max + 2, 2))
        
        # If handles are close, push text apart
        if abs(x_max - x_min) < (tw_min + tw_max + 10):
             draw_x_min = x_min - tw_min - 5
             draw_x_max = x_max + 5

        painter.drawText(int(draw_x_min), h - 5, min_str)
        painter.drawText(int(draw_x_max), h - 5, max_str)

    def mousePressEvent(self, event):
        x = event.x()
        x_min = self._val_to_x(self.clim_min)
        x_max = self._val_to_x(self.clim_max)
        
        # Hit test
        dist_min = abs(x - x_min)
        dist_max = abs(x - x_max)
        
        if dist_min < 10:
            self._dragging = 'min'
        elif dist_max < 10:
            self._dragging = 'max'
        elif x_min < x < x_max:
            self._dragging = 'center'
        else:
            self._dragging = None
            
        self._last_mouse_x = x

    def mouseMoveEvent(self, event):
        x = event.x()
        
        # Cursor updates
        x_min = self._val_to_x(self.clim_min)
        x_max = self._val_to_x(self.clim_max)
        dist_min = abs(x - x_min)
        dist_max = abs(x - x_max)
        
        if dist_min < 10 or dist_max < 10:
            self.setCursor(Qt.SizeHorCursor)
        elif x_min < x < x_max:
            self.setCursor(Qt.OpenHandCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

        if self._dragging is None:
            return

        if self._dragging == 'center':
            self.setCursor(Qt.ClosedHandCursor)
            dx_pixels = x - self._last_mouse_x
            
            # Convert pixel delta to value delta
            # We need to be careful because _x_to_val is absolute
            # Calculate span per pixel
            w = self.width()
            data_span = self.data_max - self.data_min
            if w > 0:
                val_per_pixel = data_span / w
                d_val = dx_pixels * val_per_pixel
                
                new_min = self.clim_min + d_val
                new_max = self.clim_max + d_val
                
                # Clamp to data bounds
                if new_min >= self.data_min and new_max <= self.data_max:
                    self.clim_min = new_min
                    self.clim_max = new_max
                    self.climChanged.emit(self.clim_min, self.clim_max)
                    self.update()
            
        else:
            val = self._x_to_val(x)
            if self._dragging == 'min':
                self.clim_min = min(val, self.clim_max - 1e-5)
            elif self._dragging == 'max':
                self.clim_max = max(val, self.clim_min + 1e-5)
            
            self.climChanged.emit(self.clim_min, self.clim_max)
            self.update()

        self._last_mouse_x = x

    def mouseReleaseEvent(self, event):
        self._dragging = None
        self.setCursor(Qt.ArrowCursor)


class ContrastDialog(QDialog):
    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.setWindowTitle("Brightness / Contrast")
        self.resize(400, 250)
        self.setWindowFlags(Qt.Tool)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)

        # 0. Window ID Label
        if hasattr(viewer, 'window_id'):
            wid_label = QLabel(f"<b>Window ID: {viewer.window_id}</b>")
            wid_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(wid_label)

        # 1. Channel Selector
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Channel:"))
        self.combo = QComboBox()
        self.combo.addItems([f"Channel {i + 1}" for i in range(viewer.C)])
        self.combo.currentIndexChanged.connect(self.refresh_ui)
        row1.addWidget(self.combo)
        layout.addLayout(row1)

        # 2. Interactive Histogram
        self.hist_widget = HistogramWidget()
        self.hist_widget.climChanged.connect(self.on_clim_changed)
        layout.addWidget(self.hist_widget)

        # 3. Gamma Control
        gamma_layout = QHBoxLayout()
        gamma_layout.addWidget(QLabel("Gamma:"))
        
        self.gamma_slider = QSlider(Qt.Horizontal)
        self.gamma_slider.setRange(1, 400) # 0.01 to 4.00
        self.gamma_slider.setValue(100)
        self.gamma_slider.valueChanged.connect(self.on_gamma_slider_changed)
        gamma_layout.addWidget(self.gamma_slider)
        
        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.01, 4.0)
        self.gamma_spin.setSingleStep(0.1)
        self.gamma_spin.setValue(1.0)
        self.gamma_spin.valueChanged.connect(self.on_gamma_spin_changed)
        gamma_layout.addWidget(self.gamma_spin)
        
        layout.addLayout(gamma_layout)

        # 4. Auto Button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.btn_auto = QPushButton("Auto Contrast")
        self.btn_auto.setCursor(Qt.PointingHandCursor)
        self.btn_auto.clicked.connect(self.auto_contrast)
        btn_layout.addWidget(self.btn_auto)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Initial Load
        self.refresh_ui()

    def refresh_ui(self):
        c_idx = self.combo.currentIndex()
        cache = self.viewer.renderer.current_slice_cache
        if cache is None:
            return

        if c_idx < cache.shape[0]:
            plane = cache[c_idx]
            
            # Update Histogram Data
            color = self.viewer.renderer.channel_colors[c_idx % 6]
            self.hist_widget.set_data(plane, color)

            # Get current clim from renderer
            curr_min, curr_max = self.viewer.renderer.get_clim(c_idx)
            
            # Update widget handles without triggering signal loop
            self.hist_widget.blockSignals(True)
            self.hist_widget.set_clim(curr_min, curr_max)
            self.hist_widget.blockSignals(False)
            
            # Update Gamma
            gamma = self.viewer.renderer.get_gamma(c_idx)
            self.block_gamma_signals(True)
            self.gamma_spin.setValue(gamma)
            self.gamma_slider.setValue(int(gamma * 100))
            self.block_gamma_signals(False)

    def block_gamma_signals(self, block):
        self.gamma_slider.blockSignals(block)
        self.gamma_spin.blockSignals(block)

    def on_gamma_slider_changed(self, val):
        gamma = val / 100.0
        self.gamma_spin.blockSignals(True)
        self.gamma_spin.setValue(gamma)
        self.gamma_spin.blockSignals(False)
        self.update_gamma(gamma)

    def on_gamma_spin_changed(self, val):
        self.gamma_slider.blockSignals(True)
        self.gamma_slider.setValue(int(val * 100))
        self.gamma_slider.blockSignals(False)
        self.update_gamma(val)

    def update_gamma(self, gamma):
        c_idx = self.combo.currentIndex()
        self.viewer.renderer.set_gamma(c_idx, gamma)
        self.viewer.canvas.update()

    def on_clim_changed(self, vmin, vmax):
        c_idx = self.combo.currentIndex()
        self.viewer.renderer.set_clim(c_idx, vmin, vmax)
        self.viewer.canvas.update()

    def auto_contrast(self):
        c_idx = self.combo.currentIndex()
        cache = self.viewer.renderer.current_slice_cache
        if cache is not None:
            plane = cache[c_idx]
            mn, mx = float(np.nanmin(plane)), float(np.nanmax(plane))
            
            # Update Renderer
            self.viewer.renderer.set_clim(c_idx, mn, mx)
            self.viewer.canvas.update()
            
            # Update Widget
            self.hist_widget.blockSignals(True)
            self.hist_widget.set_clim(mn, mx)
            self.hist_widget.blockSignals(False)


from qtpy.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView

class MetadataDialog(QDialog):
    def __init__(self, metadata, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Metadata")
        self.resize(400, 500)
        
        layout = QVBoxLayout(self)
        
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Key", "Value"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        
        layout.addWidget(self.table)
        
        self.populate_table(metadata)
        
    def populate_table(self, metadata):
        self.table.setRowCount(0)
        for key, value in metadata.items():
            row = self.table.rowCount()
            self.table.insertRow(row)
            
            # Key
            k_item = QTableWidgetItem(str(key))
            k_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.table.setItem(row, 0, k_item)
            
            # Value
            v_str = str(value)
            # If value is a long list/array, truncate it?
            if isinstance(value, (list, tuple, np.ndarray)) and len(value) > 10:
                v_str = f"{type(value).__name__} shape={np.shape(value)}"
                
            v_item = QTableWidgetItem(v_str)
            v_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.table.setItem(row, 1, v_item)
