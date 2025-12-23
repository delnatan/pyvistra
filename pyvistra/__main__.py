import sys

from qtpy.QtWidgets import QApplication


def main():
    # Create the Qt Application FIRST (before importing any modules that create QObjects)
    qt_app = QApplication(sys.argv)

    # Now it's safe to import modules that create QObjects at module level
    from pyvistra.ui import Toolbar
    from pyvistra.theme import DARK_THEME

    # Apply global stylesheet
    qt_app.setStyleSheet(DARK_THEME)

    # Create the floating toolbar
    toolbar = Toolbar()
    toolbar.show()

    # Execute loop
    sys.exit(qt_app.exec_())


if __name__ == "__main__":
    main()
