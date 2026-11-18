"""theme.apply: the stylesheet plus the native-chrome corrections it needs."""

import re
from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication, QCheckBox, QTabWidget, QVBoxLayout, QWidget

from pyvistra import theme

_APP = None


def _app():
    global _APP
    app = QApplication.instance()
    if app is None:
        QApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)
        app = _APP = QApplication([])  # module-level ref: a collected app aborts
    return app


def test_every_image_the_stylesheet_names_exists():
    for url in re.findall(r"url\(([^)]+)\)", theme.DARK_THEME):
        assert Path(url).is_file(), url


def test_apply_lays_check_boxes_out_by_their_painted_rect_and_drops_tab_bases():
    app = _app()
    theme.apply(app)
    theme.apply(app)  # idempotent
    host = QWidget()
    layout = QVBoxLayout(host)
    box = QCheckBox("restrict to regions")
    tabs = QTabWidget()
    tabs.setDocumentMode(True)
    tabs.addTab(QWidget(), "one")
    layout.addWidget(box)
    layout.addWidget(tabs)
    host.show()
    app.processEvents()
    assert box.testAttribute(Qt.WidgetAttribute.WA_LayoutUsesWidgetRect)
    assert not tabs.tabBar().drawBase()
    host.close()
