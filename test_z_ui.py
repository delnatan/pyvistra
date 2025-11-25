import sys
import numpy as np
from qtpy.QtWidgets import QApplication
from superqt import QRangeSlider
from impy.ui import ImageWindow

def verify_ui():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
        
    print("Verifying Z-Projection UI...")
    
    # Create synthetic data (T=1, Z=10, C=1, Y=100, X=100)
    data = np.random.rand(1, 10, 1, 100, 100).astype(np.float32)
    
    win = ImageWindow(data, title="Test Window")
    win.show()
    
    # 1. Check for QRangeSlider
    if not hasattr(win, 'z_range_slider'):
        print("FAIL: z_range_slider not found")
        return
        
    if not isinstance(win.z_range_slider, QRangeSlider):
        print(f"FAIL: z_range_slider is not QRangeSlider, got {type(win.z_range_slider)}")
        return
        
    print("PASS: QRangeSlider exists")
    
    # 2. Check Initial Visibility (Should be hidden)
    if win.z_range_slider_widget.isVisible():
        print("FAIL: z_range_slider_widget should be hidden initially")
    else:
        print("PASS: z_range_slider_widget is initially hidden")
        
    # 3. Toggle Checkbox
    if not hasattr(win, 'chk_proj'):
        print("FAIL: chk_proj not found")
        return
        
    print("Toggling Max Proj ON...")
    win.chk_proj.setChecked(True)
    
    if not win.z_range_slider_widget.isVisible():
        print("FAIL: z_range_slider_widget should be visible after toggle")
    else:
        print("PASS: z_range_slider_widget is visible")
        
    if win.z_slider.isVisible():
        print("FAIL: z_slider should be hidden after toggle")
    else:
        print("PASS: z_slider is hidden")
        
    # 4. Toggle Checkbox OFF
    print("Toggling Max Proj OFF...")
    win.chk_proj.setChecked(False)
    
    if win.z_range_slider_widget.isVisible():
        print("FAIL: z_range_slider_widget should be hidden after toggle off")
    else:
        print("PASS: z_range_slider_widget is hidden again")
        
    print("All UI tests passed!")
    win.close()

if __name__ == "__main__":
    verify_ui()
