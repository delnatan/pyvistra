import json
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QPushButton, 
    QLabel, QFileDialog, QListWidgetItem, QComboBox, QMenuBar, QAction
)
from qtpy.QtCore import Qt
from .manager import manager
from .rois import CoordinateROI, RectangleROI, CircleROI, LineROI
from .analysis import plot_profile, crop_image, measure_intensity

class ROIManager(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ROI Manager")
        self.resize(300, 400)
        self.active_window = None
        
        self.layout = QVBoxLayout(self)
        
        # Window Selection
        win_layout = QHBoxLayout()
        win_layout.addWidget(QLabel("Window:"))
        self.window_combo = QComboBox()
        self.window_combo.currentIndexChanged.connect(self.on_window_combo_changed)
        win_layout.addWidget(self.window_combo)
        self.layout.addLayout(win_layout)
        
        # List
        self.roi_list = QListWidget()
        self.roi_list.itemClicked.connect(self.on_item_clicked)
        self.layout.addWidget(self.roi_list)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.btn_delete = QPushButton("Delete")
        self.btn_delete.clicked.connect(self.delete_roi)
        btn_layout.addWidget(self.btn_delete)
        
        self.btn_save = QPushButton("Save")
        self.btn_save.clicked.connect(self.save_rois)
        btn_layout.addWidget(self.btn_save)
        
        self.btn_load = QPushButton("Load")
        self.btn_load.clicked.connect(self.load_rois)
        btn_layout.addWidget(self.btn_load)
        
        self.layout.addLayout(btn_layout)
        
        # Menu Bar
        self.menu_bar = QMenuBar()
        self.layout.setMenuBar(self.menu_bar)
        
        # Analysis Menu
        analysis_menu = self.menu_bar.addMenu("Analysis")
        
        action_profile = QAction("Plot Profile", self)
        action_profile.triggered.connect(lambda: self.run_analysis(plot_profile))
        analysis_menu.addAction(action_profile)
        
        action_crop = QAction("Crop Image", self)
        action_crop.triggered.connect(lambda: self.run_analysis(crop_image))
        analysis_menu.addAction(action_crop)
        
        action_measure = QAction("Measure Intensity", self)
        action_measure.triggered.connect(lambda: self.run_analysis(measure_intensity))
        analysis_menu.addAction(action_measure)

    def refresh_windows(self):
        """Populate the window combo box."""
        self.window_combo.blockSignals(True)
        self.window_combo.clear()
        
        windows = manager.get_all()
        for wid, win in windows.items():
            title = win.windowTitle()
            self.window_combo.addItem(title, userData=wid)
            
        # Select active if present
        if self.active_window:
            idx = self.window_combo.findData(self.active_window.window_id)
            if idx >= 0:
                self.window_combo.setCurrentIndex(idx)
                
        self.window_combo.blockSignals(False)

    def on_window_combo_changed(self, index):
        if index < 0:
            return
        wid = self.window_combo.itemData(index)
        win = manager.get(wid)
        if win:
            self.set_active_window(win)

    def set_active_window(self, window):
        if self.active_window == window:
            return
            
        self.active_window = window
        self.setWindowTitle(f"ROI Manager - Window {window.window_id}")
        self.refresh_windows() # Ensure combo is up to date and selected
        self.refresh_list()

    def refresh_list(self):
        self.roi_list.clear()
        if not self.active_window:
            return
            
        for i, roi in enumerate(self.active_window.rois):
            item = QListWidgetItem(f"{i}: {roi.name} ({roi.__class__.__name__})")
            item.setData(Qt.UserRole, roi)
            self.roi_list.addItem(item)

    def add_roi(self, roi):
        # Called by ImageWindow when new ROI is drawn
        self.refresh_list()

    def delete_roi(self):
        item = self.roi_list.currentItem()
        if not item or not self.active_window:
            return
            
        roi = item.data(Qt.UserRole)
        roi.remove()
        self.active_window.rois.remove(roi)
        self.refresh_list()
        self.active_window.canvas.update()

    def on_item_clicked(self, item):
        # Highlight logic could go here
        pass

    def save_rois(self):
        if not self.active_window:
            return
            
        path, _ = QFileDialog.getSaveFileName(self, "Save ROIs", ".", "JSON Files (*.json)")
        if not path:
            return
            
        data = [roi.to_dict() for roi in self.active_window.rois]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_rois(self):
        if not self.active_window:
            return
            
        path, _ = QFileDialog.getOpenFileName(self, "Load ROIs", ".", "JSON Files (*.json)")
        if not path:
            return
            
        with open(path, "r") as f:
            data = json.load(f)
            
        for item in data:
            cls_name = item["type"]
            if cls_name == "CoordinateROI":
                roi = CoordinateROI(self.active_window.view, name=item["name"])
            elif cls_name == "RectangleROI":
                roi = RectangleROI(self.active_window.view, name=item["name"])
            elif cls_name == "CircleROI":
                roi = CircleROI(self.active_window.view, name=item["name"])
            elif cls_name == "LineROI":
                roi = LineROI(self.active_window.view, name=item["name"])
            else:
                continue
                
            roi.from_dict(item["data"])
            self.active_window.rois.append(roi)
            
        self.refresh_list()
        self.active_window.canvas.update()

    def run_analysis(self, func):
        item = self.roi_list.currentItem()
        if not item or not self.active_window:
            print("No ROI selected")
            return
            
        roi = item.data(Qt.UserRole)
        
        # Prepare Data
        # For profile/measure, we usually want the CURRENT slice (2D)
        # For crop, we might want 5D?
        # Let's check the function signature or just pass what makes sense.
        # The analysis functions currently handle 2D or 5D checks.
        
        # Get 5D data
        data_5d = self.active_window.img_data
        
        # Get 2D slice
        # We need to access the cache or slice it manually using window indices
        t = self.active_window.t_idx
        z = self.active_window.z_idx
        # Handle Z-slice (projection) if active?
        # If projection is active, z is a slice.
        # The proxy handles it.
        
        # But wait, `img_data` is the proxy or array.
        # If we want the VISIBLE image (e.g. projected), we should slice it.
        # If we pass the whole 5D proxy to `crop`, it works.
        # If we pass 5D to `plot_profile`, it fails (expects 2D).
        
        # Let's try to pass the appropriate data.
        if func == crop_image:
            # Pass full data
            func(data_5d, roi)
        else:
            # Pass current slice (2D)
            # We can use the renderer's cache if available, or slice the proxy.
            # Renderer cache is (C, Y, X) or (Y, X).
            # If Composite, it's (C, Y, X). Profile needs 2D.
            # Let's use the active channel if composite.
            
            # Slice manually to be safe
            # We need to handle the Z-slice logic from UI?
            # The UI constructs a slice for Z if projection is on.
            # But here we don't have easy access to that logic without duplicating it.
            # Let's just use the current z_idx (int) for now, or ask the window?
            
            # Actually, let's just grab what's in the renderer cache?
            cache = self.active_window.renderer.current_slice_cache
            if cache is None:
                print("No image data available")
                return
                
            # cache is (C, Y, X) or (Y, X)
            if cache.ndim == 3:
                # Use active channel
                c = self.active_window.c_idx
                if c < cache.shape[0]:
                    data_2d = cache[c]
                else:
                    data_2d = cache[0] # Fallback
            else:
                data_2d = cache
                
            func(data_2d, roi)

# Global instance
_roi_manager_instance = None

def get_roi_manager():
    global _roi_manager_instance
    if _roi_manager_instance is None:
        _roi_manager_instance = ROIManager()
    return _roi_manager_instance


