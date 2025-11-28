import sys
import faulthandler
from qtpy.QtWidgets import QApplication, QWidget
from qtpy.QtCore import QTimer

# Enable faulthandler to dump traceback on segfault
faulthandler.enable()

from impy.ui import Toolbar
from impy.styles import DARK_THEME

def run_test():
    # Import Vispy FIRST
    import vispy
    # Force backend if possible?
    # vispy.use('pyqt6') # Not standard API
    from vispy import app as vispy_app
    vispy_app.use_app('pyqt6')
    print("Imported Vispy and set backend")

    print("Starting minimal test...")
    app = QApplication(sys.argv)
    
    # Create a dummy widget to ensure Qt is active
    w = QWidget()
    w.show()
    
    def quit_app():
        print("Quitting...")
        w.close()
        app.quit()
        
    QTimer.singleShot(1000, quit_app)
    
    app.exec_()
    print("Finished.")

if __name__ == "__main__":
    run_test()
