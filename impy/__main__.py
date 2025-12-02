import sys

from qtpy.QtWidgets import QApplication

from impy.ui import Toolbar


def main():
    # Create the Qt Application (agnostic backend)
    qt_app = QApplication(sys.argv)

    # Apply global stylesheet
    from impy.theme import DARK_THEME
    qt_app.setStyleSheet(DARK_THEME)

    # Create the floating toolbar
    toolbar = Toolbar()
    toolbar.show()

    # Execute loop
    sys.exit(qt_app.exec_())


if __name__ == "__main__":
    main()
