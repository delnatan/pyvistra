import sys
import numpy as np
from qtpy.QtWidgets import QApplication
from impy.ui import ImageWindow
from impy.rois import RectangleROI

def reproduce_error():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
        
    print("Reproducing ROI Error...")
    
    data = np.zeros((1, 1, 1, 100, 100), dtype=np.uint8)
    win = ImageWindow(data, title="Repro Test")
    # Don't show window to avoid blocking, just test logic
    
    rect = RectangleROI(win.view)
    
    try:
        # Simulate initial click where p1 == p2
        print("Attempting update with zero size...")
        rect.update((10, 10), (10, 10))
        print("FAIL: No error raised")
    except ValueError as e:
        print(f"PASS: Caught expected error: {e}")
    except Exception as e:
        print(f"PASS: Caught unexpected error: {e}")

if __name__ == "__main__":
    reproduce_error()
