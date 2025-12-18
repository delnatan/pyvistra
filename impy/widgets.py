import numpy as np
from qtpy.QtCore import Qt, Signal, QRectF
from qtpy.QtGui import QBrush, QColor, QFont, QPainter, QPen
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QSlider,
    QDoubleSpinBox,
    QGroupBox,
)
from impy.visuals import COLORMAPS

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
        x = int(ratio * w)
        # Clamp to 32-bit signed integer range to prevent Qt OverflowError
        return max(-2147483648, min(x, 2147483647))

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
        self.resize(450, 280)
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

        # 1b. Colormap Selector
        row_cmap = QHBoxLayout()
        row_cmap.addWidget(QLabel("Colormap:"))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(list(COLORMAPS.keys()))
        self.cmap_combo.currentTextChanged.connect(self.on_colormap_changed)
        row_cmap.addWidget(self.cmap_combo)
        layout.addLayout(row_cmap)

        # 2. Interactive Histogram with Min/Max Spinboxes
        hist_layout = QHBoxLayout()
        hist_layout.setSpacing(6)

        # Min spinbox
        min_label = QLabel("Min:")
        min_label.setStyleSheet("color: #AAA; font-size: 10px;")
        hist_layout.addWidget(min_label)

        self.min_spin = QDoubleSpinBox()
        self.min_spin.setDecimals(1)
        self.min_spin.setRange(-1e9, 1e9)
        self.min_spin.setSingleStep(10)
        self.min_spin.setFixedWidth(75)
        self.min_spin.setToolTip("Minimum intensity")
        self.min_spin.valueChanged.connect(self.on_min_spin_changed)
        hist_layout.addWidget(self.min_spin)

        self.hist_widget = HistogramWidget()
        self.hist_widget.climChanged.connect(self.on_histogram_clim_changed)
        hist_layout.addWidget(self.hist_widget, 1)

        # Max spinbox
        max_label = QLabel("Max:")
        max_label.setStyleSheet("color: #AAA; font-size: 10px;")
        hist_layout.addWidget(max_label)

        self.max_spin = QDoubleSpinBox()
        self.max_spin.setDecimals(1)
        self.max_spin.setRange(-1e9, 1e9)
        self.max_spin.setSingleStep(10)
        self.max_spin.setFixedWidth(75)
        self.max_spin.setToolTip("Maximum intensity")
        self.max_spin.valueChanged.connect(self.on_max_spin_changed)
        hist_layout.addWidget(self.max_spin)

        layout.addLayout(hist_layout)

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

        # 4. Auto/Manual Contrast Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        self.btn_loosen = QPushButton("-")
        self.btn_loosen.setToolTip("Loosen Contrast (Expand Range)")
        self.btn_loosen.setFixedWidth(30)
        self.btn_loosen.clicked.connect(lambda: self.adjust_contrast(-1))
        btn_layout.addWidget(self.btn_loosen)

        self.btn_auto = QPushButton("Auto Contrast")
        self.btn_auto.setCursor(Qt.PointingHandCursor)
        self.btn_auto.clicked.connect(self.reset_auto_contrast)
        btn_layout.addWidget(self.btn_auto)

        self.btn_tighten = QPushButton("+")
        self.btn_tighten.setToolTip("Tighten Contrast (Shrink Range)")
        self.btn_tighten.setFixedWidth(30)
        self.btn_tighten.clicked.connect(lambda: self.adjust_contrast(1))
        btn_layout.addWidget(self.btn_tighten)

        self.chk_all_channels = QCheckBox("All Channels")
        self.chk_all_channels.setChecked(True)
        btn_layout.addWidget(self.chk_all_channels)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # State for auto-contrast
        self.pct_low = 0.2
        self.pct_high = 99.98

        # Initial Load
        self.refresh_ui()

    def refresh_ui(self):
        c_idx = self.combo.currentIndex()
        cache = self.viewer.renderer.current_slice_cache
        if cache is None:
            return

        if c_idx < cache.shape[0]:
            plane = cache[c_idx]

            # Update Colormap Dropdown
            cmap_name = self.viewer.renderer.get_colormap_name(c_idx)
            self.cmap_combo.blockSignals(True)
            idx = self.cmap_combo.findText(cmap_name)
            if idx >= 0:
                self.cmap_combo.setCurrentIndex(idx)
            self.cmap_combo.blockSignals(False)

            # Update Histogram Data
            color = self.viewer.renderer.channel_colors[c_idx % 6]
            self.hist_widget.set_data(plane, color)

            # Get current clim from renderer
            curr_min, curr_max = self.viewer.renderer.get_clim(c_idx)

            # Update histogram and spinboxes without triggering signal loop
            self.block_clim_signals(True)
            self.hist_widget.set_clim(curr_min, curr_max)
            self.min_spin.setValue(curr_min)
            self.max_spin.setValue(curr_max)
            self.block_clim_signals(False)

            # Update Gamma
            gamma = self.viewer.renderer.get_gamma(c_idx)
            self.block_gamma_signals(True)
            self.gamma_spin.setValue(gamma)
            self.gamma_slider.setValue(int(gamma * 100))
            self.block_gamma_signals(False)

    def block_clim_signals(self, block):
        """Block or unblock signals from clim-related widgets."""
        self.hist_widget.blockSignals(block)
        self.min_spin.blockSignals(block)
        self.max_spin.blockSignals(block)

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

    def on_histogram_clim_changed(self, vmin, vmax):
        """Handle histogram clim change (from dragging handles)."""
        # Update spinboxes
        self.min_spin.blockSignals(True)
        self.max_spin.blockSignals(True)
        self.min_spin.setValue(vmin)
        self.max_spin.setValue(vmax)
        self.min_spin.blockSignals(False)
        self.max_spin.blockSignals(False)
        # Update renderer
        c_idx = self.combo.currentIndex()
        self.viewer.renderer.set_clim(c_idx, vmin, vmax)
        self.viewer.canvas.update()

    def on_min_spin_changed(self, value):
        """Handle min spinbox change."""
        max_val = self.max_spin.value()
        if value < max_val:
            # Update histogram
            self.hist_widget.blockSignals(True)
            self.hist_widget.set_clim(value, max_val)
            self.hist_widget.blockSignals(False)
            # Update renderer
            c_idx = self.combo.currentIndex()
            self.viewer.renderer.set_clim(c_idx, value, max_val)
            self.viewer.canvas.update()

    def on_max_spin_changed(self, value):
        """Handle max spinbox change."""
        min_val = self.min_spin.value()
        if value > min_val:
            # Update histogram
            self.hist_widget.blockSignals(True)
            self.hist_widget.set_clim(min_val, value)
            self.hist_widget.blockSignals(False)
            # Update renderer
            c_idx = self.combo.currentIndex()
            self.viewer.renderer.set_clim(c_idx, min_val, value)
            self.viewer.canvas.update()

    def on_colormap_changed(self, cmap_name):
        c_idx = self.combo.currentIndex()
        self.viewer.renderer.set_colormap(c_idx, cmap_name)
        self.viewer.canvas.update()

        # Update histogram color to match new colormap
        color = self.viewer.renderer.channel_colors[c_idx % 6]
        cache = self.viewer.renderer.current_slice_cache
        if cache is not None and c_idx < cache.shape[0]:
            self.hist_widget.set_data(cache[c_idx], color)

    def reset_auto_contrast(self):
        """Reset to default robust percentiles."""
        self.pct_low = 0.5
        self.pct_high = 99.98
        self.apply_auto_contrast()

    def adjust_contrast(self, direction):
        """
        Adjust percentiles to tighten (+1) or loosen (-1) contrast.
        Step size: 0.01%
        """
        step = 0.01
        
        if direction > 0: # Tighten
            self.pct_low += step
            self.pct_high -= step
        else: # Loosen
            self.pct_low -= step
            self.pct_high += step
            
        # Clamp
        self.pct_low = max(0.0, min(self.pct_low, 49.0))
        self.pct_high = max(51.0, min(self.pct_high, 100.0))
        
        self.apply_auto_contrast()

    def apply_auto_contrast(self):
        c_idx = self.combo.currentIndex()
        cache = self.viewer.renderer.current_slice_cache

        if self.chk_all_channels.isChecked():
            # Apply to all channels
            for ch_idx in range(self.combo.count()):
                plane = cache[ch_idx]
                # Ignore zeros (background)
                valid_data = plane[plane > 0]
                if valid_data.size == 0:
                    valid_data = plane  # Fallback if all zeros

                mn, mx = map(float, np.nanpercentile(valid_data, (self.pct_low, self.pct_high)))

                # Update Renderer
                self.viewer.renderer.set_clim(ch_idx, mn, mx)

            # Update widgets for currently selected channel
            curr_min, curr_max = self.viewer.renderer.get_clim(c_idx)
            self.block_clim_signals(True)
            self.hist_widget.set_clim(curr_min, curr_max)
            self.min_spin.setValue(curr_min)
            self.max_spin.setValue(curr_max)
            self.block_clim_signals(False)
            self.viewer.canvas.update()
            return

        if cache is not None:
            plane = cache[c_idx]
            # Ignore zeros (background)
            valid_data = plane[plane > 0]
            if valid_data.size == 0:
                valid_data = plane  # Fallback if all zeros

            mn, mx = map(float, np.nanpercentile(valid_data, (self.pct_low, self.pct_high)))

            # Update Renderer
            self.viewer.renderer.set_clim(c_idx, mn, mx)
            self.viewer.canvas.update()

            # Update widgets
            self.block_clim_signals(True)
            self.hist_widget.set_clim(mn, mx)
            self.min_spin.setValue(mn)
            self.max_spin.setValue(mx)
            self.block_clim_signals(False)


from qtpy.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView, QScrollArea, QFrame, QSizePolicy


class CompactHistogramWidget(QWidget):
    """
    A compact version of HistogramWidget for use in stacked channel panels.
    Smaller height, no text labels, focused on visual feedback.
    """

    climChanged = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(40)
        self.setMaximumHeight(50)
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
        if span <= 0:
            return 0
        ratio = (val - self.data_min) / span
        x = int(ratio * w)
        return max(-2147483648, min(x, 2147483647))

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

        # Draw Histogram
        if self.hist_data is not None:
            max_log = np.max(self.hist_data)
            if max_log == 0:
                max_log = 1

            fill_color = QColor(self.color)
            fill_color.setAlpha(100)
            painter.setBrush(QBrush(fill_color))
            painter.setPen(Qt.NoPen)

            n_bins = len(self.hist_data)
            bin_w = w / n_bins

            for i, val in enumerate(self.hist_data):
                bar_h = (val / max_log) * (h - 4)
                x = i * bin_w
                y = h - bar_h
                painter.drawRect(QRectF(x, y, bin_w, bar_h))

        # Draw Overlay (Darken outside selection)
        x_min = self._val_to_x(self.clim_min)
        x_max = self._val_to_x(self.clim_max)

        dark_overlay = QColor(0, 0, 0, 150)
        painter.fillRect(0, 0, x_min, h, dark_overlay)
        painter.fillRect(x_max, 0, w - x_max, h, dark_overlay)

        # Draw Handles (thinner for compact view)
        pen = QPen(HANDLE_COLOR)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawLine(x_min, 0, x_min, h)
        painter.drawLine(x_max, 0, x_max, h)

    def mousePressEvent(self, event):
        x = event.x()
        x_min = self._val_to_x(self.clim_min)
        x_max = self._val_to_x(self.clim_max)

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

            w = self.width()
            data_span = self.data_max - self.data_min
            if w > 0:
                val_per_pixel = data_span / w
                d_val = dx_pixels * val_per_pixel

                new_min = self.clim_min + d_val
                new_max = self.clim_max + d_val

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


class ChannelRow(QWidget):
    """
    A single row representing one channel with:
    - Visibility checkbox
    - Color swatch (clickable for colormap selection)
    - Channel name label
    - Min/Max spinboxes for intensity range
    - Compact histogram
    - Gamma adjustment
    """

    visibilityChanged = Signal(int, bool)  # channel_idx, visible
    climChanged = Signal(int, float, float)  # channel_idx, vmin, vmax
    colormapChanged = Signal(int, str)  # channel_idx, colormap_name
    gammaChanged = Signal(int, float)  # channel_idx, gamma

    def __init__(self, channel_idx, channel_name, color, parent=None):
        super().__init__(parent)
        self.channel_idx = channel_idx

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)

        # Visibility checkbox
        self.chk_visible = QCheckBox()
        self.chk_visible.setChecked(True)
        self.chk_visible.setToolTip("Toggle channel visibility")
        self.chk_visible.toggled.connect(self._on_visibility_changed)
        layout.addWidget(self.chk_visible)

        # Color swatch (button that opens colormap menu)
        self.color_btn = QPushButton()
        self.color_btn.setFixedSize(20, 20)
        self.color_btn.setCursor(Qt.PointingHandCursor)
        self.color_btn.setToolTip("Change colormap")
        self._update_color_swatch(color)
        self.color_btn.clicked.connect(self._show_colormap_menu)
        layout.addWidget(self.color_btn)

        # Channel name label
        self.name_label = QLabel(channel_name)
        self.name_label.setFixedWidth(40)
        self.name_label.setStyleSheet("color: #EEE; font-size: 11px;")
        self.name_label.setToolTip(channel_name)  # Show full name on hover
        layout.addWidget(self.name_label)

        # Min spinbox for contrast
        self.min_spin = QDoubleSpinBox()
        self.min_spin.setDecimals(1)
        self.min_spin.setRange(-1e9, 1e9)
        self.min_spin.setSingleStep(10)
        self.min_spin.setFixedWidth(65)
        self.min_spin.setToolTip("Minimum intensity")
        self.min_spin.valueChanged.connect(self._on_min_changed)
        layout.addWidget(self.min_spin)

        # Compact histogram
        self.histogram = CompactHistogramWidget()
        self.histogram.climChanged.connect(self._on_histogram_clim_changed)
        layout.addWidget(self.histogram, 1)

        # Max spinbox for contrast
        self.max_spin = QDoubleSpinBox()
        self.max_spin.setDecimals(1)
        self.max_spin.setRange(-1e9, 1e9)
        self.max_spin.setSingleStep(10)
        self.max_spin.setFixedWidth(65)
        self.max_spin.setToolTip("Maximum intensity")
        self.max_spin.valueChanged.connect(self._on_max_changed)
        layout.addWidget(self.max_spin)

        # Gamma spinbox
        gamma_label = QLabel("γ")
        gamma_label.setStyleSheet("color: #AAA; font-size: 10px;")
        gamma_label.setFixedWidth(10)
        layout.addWidget(gamma_label)

        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.1, 4.0)
        self.gamma_spin.setSingleStep(0.1)
        self.gamma_spin.setValue(1.0)
        self.gamma_spin.setFixedWidth(50)
        self.gamma_spin.setToolTip("Gamma correction")
        self.gamma_spin.valueChanged.connect(self._on_gamma_changed)
        layout.addWidget(self.gamma_spin)

        self.current_colormap = "White"

    def _update_color_swatch(self, color):
        """Update the color swatch button background."""
        self.color_btn.setStyleSheet(
            f"background-color: {color}; border: 1px solid #555; border-radius: 3px;"
        )

    def _on_visibility_changed(self, checked):
        self.visibilityChanged.emit(self.channel_idx, checked)

    def _on_min_changed(self, value):
        """Handle min spinbox change."""
        max_val = self.max_spin.value()
        if value < max_val:
            self.climChanged.emit(self.channel_idx, value, max_val)
            # Update histogram display
            self.histogram.blockSignals(True)
            self.histogram.set_clim(value, max_val)
            self.histogram.blockSignals(False)

    def _on_max_changed(self, value):
        """Handle max spinbox change."""
        min_val = self.min_spin.value()
        if value > min_val:
            self.climChanged.emit(self.channel_idx, min_val, value)
            # Update histogram display
            self.histogram.blockSignals(True)
            self.histogram.set_clim(min_val, value)
            self.histogram.blockSignals(False)

    def _on_histogram_clim_changed(self, vmin, vmax):
        """Handle histogram clim change (from dragging handles)."""
        # Update spinboxes
        self.min_spin.blockSignals(True)
        self.max_spin.blockSignals(True)
        self.min_spin.setValue(vmin)
        self.max_spin.setValue(vmax)
        self.min_spin.blockSignals(False)
        self.max_spin.blockSignals(False)
        # Emit signal to parent
        self.climChanged.emit(self.channel_idx, vmin, vmax)

    def _on_gamma_changed(self, value):
        self.gammaChanged.emit(self.channel_idx, value)

    def _show_colormap_menu(self):
        """Show a popup menu for colormap selection."""
        from qtpy.QtWidgets import QMenu
        menu = QMenu(self)

        for cmap_name in COLORMAPS.keys():
            action = menu.addAction(cmap_name)
            action.triggered.connect(lambda checked, name=cmap_name: self._on_colormap_selected(name))

        menu.exec_(self.color_btn.mapToGlobal(self.color_btn.rect().bottomLeft()))

    def _on_colormap_selected(self, cmap_name):
        self.current_colormap = cmap_name
        self.colormapChanged.emit(self.channel_idx, cmap_name)

    def set_data(self, data_slice, color):
        """Update histogram data and color."""
        self._update_color_swatch(color)
        self.histogram.set_data(data_slice, color)

    def set_clim(self, vmin, vmax):
        """Update contrast limits display (histogram and spinboxes)."""
        self.histogram.blockSignals(True)
        self.min_spin.blockSignals(True)
        self.max_spin.blockSignals(True)
        self.histogram.set_clim(vmin, vmax)
        self.min_spin.setValue(vmin)
        self.max_spin.setValue(vmax)
        self.histogram.blockSignals(False)
        self.min_spin.blockSignals(False)
        self.max_spin.blockSignals(False)

    def set_visible_state(self, visible):
        """Update checkbox state without emitting signal."""
        self.chk_visible.blockSignals(True)
        self.chk_visible.setChecked(visible)
        self.chk_visible.blockSignals(False)

    def set_gamma(self, gamma):
        """Update gamma spinbox without emitting signal."""
        self.gamma_spin.blockSignals(True)
        self.gamma_spin.setValue(gamma)
        self.gamma_spin.blockSignals(False)


class ChannelPanel(QDialog):
    """
    Floating dialog that displays all channels stacked vertically,
    each with visibility toggle, colormap selector, histogram, and intensity spinboxes.
    """

    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.setWindowTitle("Channels")
        self.setWindowFlags(Qt.Tool)
        self.resize(480, min(200 + viewer.C * 60, 500))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # Window ID Label
        if hasattr(viewer, 'window_id'):
            wid_label = QLabel(f"<b>Window: {viewer.window_id}</b>")
            wid_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(wid_label)

        # Scroll area for channel rows
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        scroll_content = QWidget()
        self.rows_layout = QVBoxLayout(scroll_content)
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_layout.setSpacing(2)

        scroll.setWidget(scroll_content)
        layout.addWidget(scroll, 1)

        # Channel rows
        self.channel_rows = []
        self._setup_channel_rows()

        # Auto-contrast button row
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        btn_auto = QPushButton("Auto Contrast All")
        btn_auto.clicked.connect(self._auto_contrast_all)
        btn_layout.addWidget(btn_auto)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Initial data load
        self.refresh_ui()

    def _setup_channel_rows(self):
        """Create a row widget for each channel."""
        n_channels = self.viewer.C
        meta_channels = self.viewer.meta.get("channels", [])

        for c in range(n_channels):
            # Get channel name from metadata or use default
            if c < len(meta_channels) and "name" in meta_channels[c]:
                ch_name = meta_channels[c]["name"]
            else:
                ch_name = f"Ch {c + 1}"

            # Get color from renderer
            color = self.viewer.renderer.channel_colors[c % len(self.viewer.renderer.channel_colors)]

            row = ChannelRow(c, ch_name, color)
            row.visibilityChanged.connect(self._on_visibility_changed)
            row.climChanged.connect(self._on_clim_changed)
            row.colormapChanged.connect(self._on_colormap_changed)
            row.gammaChanged.connect(self._on_gamma_changed)

            self.channel_rows.append(row)
            self.rows_layout.addWidget(row)

        self.rows_layout.addStretch()

    def _on_visibility_changed(self, channel_idx, visible):
        """Handle visibility toggle for a channel."""
        self.viewer.renderer.set_channel_visible(channel_idx, visible)
        self.viewer.canvas.update()

    def _on_clim_changed(self, channel_idx, vmin, vmax):
        """Handle contrast change for a channel."""
        self.viewer.renderer.set_clim(channel_idx, vmin, vmax)
        self.viewer.canvas.update()

    def _on_colormap_changed(self, channel_idx, cmap_name):
        """Handle colormap change for a channel."""
        self.viewer.renderer.set_colormap(channel_idx, cmap_name)
        self.viewer.canvas.update()

        # Update color swatch
        color = self.viewer.renderer.channel_colors[channel_idx % len(self.viewer.renderer.channel_colors)]
        self.channel_rows[channel_idx]._update_color_swatch(color)

        # Refresh histogram with new color
        self.refresh_ui()

    def _on_gamma_changed(self, channel_idx, gamma):
        """Handle gamma change for a channel."""
        self.viewer.renderer.set_gamma(channel_idx, gamma)
        self.viewer.canvas.update()

    def _auto_contrast_all(self):
        """Apply auto contrast to all channels."""
        cache = self.viewer.renderer.current_slice_cache
        if cache is None:
            return

        for c in range(len(self.channel_rows)):
            if c < cache.shape[0]:
                plane = cache[c]
                valid_data = plane[plane > 0]
                if valid_data.size == 0:
                    valid_data = plane

                mn, mx = map(float, np.nanpercentile(valid_data, (0.5, 99.98)))
                self.viewer.renderer.set_clim(c, mn, mx)
                self.channel_rows[c].set_clim(mn, mx)

        self.viewer.canvas.update()

    def refresh_ui(self):
        """Refresh all channel rows with current data."""
        cache = self.viewer.renderer.current_slice_cache
        if cache is None:
            return

        for c, row in enumerate(self.channel_rows):
            if c < cache.shape[0]:
                plane = cache[c]
                color = self.viewer.renderer.channel_colors[c % len(self.viewer.renderer.channel_colors)]
                row.set_data(plane, color)

                # Update clim
                vmin, vmax = self.viewer.renderer.get_clim(c)
                row.set_clim(vmin, vmax)

                # Update visibility state
                visible = self.viewer.renderer.get_channel_visible(c)
                row.set_visible_state(visible)

                # Update gamma
                gamma = self.viewer.renderer.get_gamma(c)
                row.set_gamma(gamma)


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
