"""Global Qt stylesheet: flat, hard-edged, dark ("Graphite & Amber").

Built entirely from pyvistra.colors tokens so the palette has one
source of truth. Covers every widget class pyvistra actually uses,
including ones the previous stylesheet left unstyled (QCheckBox,
QRadioButton, QDialog, QMessageBox, QProgressBar, QDockWidget,
QStatusBar, QHeaderView, QSplitter) — those otherwise fall back to
native per-OS chrome, which is what breaks cross-platform consistency.
"""

from pathlib import Path

from qtpy.QtCore import QEvent, QObject, Qt
from qtpy.QtWidgets import QCheckBox, QRadioButton, QTabBar

from . import colors as c

# Arrow glyphs for spin boxes and combo boxes (TEXT_SECONDARY, and
# TEXT_DISABLED for the "-disabled" variants). Styling these widgets at all
# switches Qt to drawing their sub-controls from the stylesheet, so without
# an image here the arrows are simply blank.
_ICONS = (Path(__file__).parent / "_resources" / "icons").as_posix()

DARK_THEME = f"""
/* Main Window & Background */
QMainWindow, QWidget {{
    background-color: {c.BG_BASE};
    color: {c.TEXT_PRIMARY};
    font-size: 12px;
}}

/* Tooltips */
QToolTip {{
    background-color: {c.BG_ELEVATED};
    color: {c.TEXT_PRIMARY};
    border: 1px solid {c.BORDER};
    border-radius: {c.RADIUS};
    padding: 4px;
}}

/* Buttons */
QPushButton, QToolButton {{
    background-color: {c.ACCENT};
    color: {c.ON_ACCENT};
    border: none;
    padding: 6px 12px;
    border-radius: {c.RADIUS};
    font-weight: 600;
}}

QPushButton:hover, QToolButton:hover {{
    background-color: {c.ACCENT_HOVER};
}}

QPushButton:pressed, QToolButton:pressed {{
    background-color: {c.ACCENT_PRESSED};
}}

QPushButton:disabled, QToolButton:disabled {{
    background-color: {c.BG_ELEVATED};
    color: {c.TEXT_DISABLED};
}}

QToolButton:checked {{
    background-color: {c.ACCENT_PRESSED};
}}

/* Line Edit & Text Inputs */
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox {{
    background-color: {c.BG_ELEVATED};
    color: {c.TEXT_PRIMARY};
    border: 1px solid {c.BORDER};
    border-radius: {c.RADIUS};
    padding: 4px;
    selection-background-color: {c.ACCENT};
    selection-color: {c.ON_ACCENT};
}}

QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus,
QSpinBox:focus, QDoubleSpinBox:focus {{
    border: 1px solid {c.BORDER_FOCUS};
}}

QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
    color: {c.TEXT_DISABLED};
}}

QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border;
    width: 16px;
    border: none;
    border-left: 1px solid {c.BORDER};
    background: transparent;
}}

QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-position: top right;
    border-top-right-radius: {c.RADIUS};
}}

QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-position: bottom right;
    border-bottom-right-radius: {c.RADIUS};
}}

QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background-color: {c.BG_ELEVATED_HOVER};
}}

QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    image: url({_ICONS}/chevron-up.svg);
    width: 8px;
    height: 8px;
}}

QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image: url({_ICONS}/chevron-down.svg);
    width: 8px;
    height: 8px;
}}

/* ":off" is a step button at its limit (the value is already the max/min). */
QSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:disabled,
QSpinBox::up-arrow:off, QDoubleSpinBox::up-arrow:off {{
    image: url({_ICONS}/chevron-up-disabled.svg);
}}

QSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:disabled,
QSpinBox::down-arrow:off, QDoubleSpinBox::down-arrow:off {{
    image: url({_ICONS}/chevron-down-disabled.svg);
}}

/* ComboBox */
QComboBox {{
    background-color: {c.BG_ELEVATED};
    color: {c.TEXT_PRIMARY};
    border: 1px solid {c.BORDER};
    border-radius: {c.RADIUS};
    padding: 4px;
    min-width: 6em;
}}

QComboBox:focus {{
    border: 1px solid {c.BORDER_FOCUS};
}}

QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 16px;
    border-left: 1px solid {c.BORDER};
}}

QComboBox::down-arrow {{
    image: url({_ICONS}/chevron-down.svg);
    width: 8px;
    height: 8px;
}}

QComboBox::down-arrow:disabled {{
    image: url({_ICONS}/chevron-down-disabled.svg);
}}

QComboBox QAbstractItemView {{
    background-color: {c.BG_ELEVATED};
    color: {c.TEXT_PRIMARY};
    selection-background-color: {c.ACCENT};
    selection-color: {c.ON_ACCENT};
    border: 1px solid {c.BORDER};
    outline: none;
}}

/* CheckBox */
QCheckBox {{
    color: {c.TEXT_PRIMARY};
    spacing: 6px;
}}

QCheckBox::indicator {{
    width: 13px;
    height: 13px;
    border: 1px solid {c.BORDER};
    border-radius: {c.RADIUS};
    background-color: {c.BG_ELEVATED};
}}

QCheckBox::indicator:hover {{
    border-color: {c.TEXT_SECONDARY};
}}

QCheckBox::indicator:checked {{
    background-color: {c.ACCENT};
    border-color: {c.ACCENT};
}}

QCheckBox:disabled {{
    color: {c.TEXT_DISABLED};
}}

QCheckBox::indicator:disabled {{
    border-color: {c.BORDER};
    background-color: {c.BG_SURFACE};
}}

/* RadioButton — circular by convention (the shape itself signals
   "choose one of many"), unlike the rectangular flat-edge language
   used everywhere else. */
QRadioButton {{
    color: {c.TEXT_PRIMARY};
    spacing: 6px;
}}

QRadioButton::indicator {{
    width: 13px;
    height: 13px;
    border: 1px solid {c.BORDER};
    border-radius: 7px;
    background-color: {c.BG_ELEVATED};
}}

QRadioButton::indicator:hover {{
    border-color: {c.TEXT_SECONDARY};
}}

QRadioButton::indicator:checked {{
    background-color: {c.ACCENT};
    border: 3px solid {c.BG_ELEVATED};
    outline: 1px solid {c.ACCENT};
}}

QRadioButton:disabled {{
    color: {c.TEXT_DISABLED};
}}

/* Lists & Trees */
QListWidget, QTreeWidget, QTableWidget {{
    background-color: {c.BG_ELEVATED};
    alternate-background-color: {c.BG_SURFACE};
    border: 1px solid {c.BORDER};
    border-radius: {c.RADIUS};
    color: {c.TEXT_PRIMARY};
    outline: none;
}}

QListWidget::item:selected, QTreeWidget::item:selected, QTableWidget::item:selected {{
    background-color: {c.ACCENT};
    color: {c.ON_ACCENT};
}}

QListWidget::item:hover, QTreeWidget::item:hover, QTableWidget::item:hover {{
    background-color: {c.BG_ELEVATED_HOVER};
}}

QHeaderView::section {{
    background-color: {c.BG_SURFACE};
    color: {c.TEXT_SECONDARY};
    border: none;
    border-right: 1px solid {c.BORDER};
    border-bottom: 1px solid {c.BORDER};
    padding: 4px 6px;
}}

QHeaderView::section:hover {{
    background-color: {c.BG_ELEVATED_HOVER};
}}

/* Sliders — flat rectangular handle, not a rounded pill/circle. */
QSlider {{
    background-color: none;
}}

QSlider::groove:horizontal {{
    border: 1px solid {c.BORDER};
    height: 4px;
    background: {c.BG_ELEVATED};
    margin: 2px 0;
    border-radius: {c.RADIUS};
}}

QSlider::handle:horizontal {{
    background: {c.TEXT_SECONDARY};
    border: 1px solid {c.TEXT_SECONDARY};
    width: 12px;
    height: 14px;
    margin: -6px 0;
    border-radius: {c.RADIUS};
}}

QSlider::handle:horizontal:hover {{
    background: {c.TEXT_PRIMARY};
    border-color: {c.TEXT_PRIMARY};
}}

QSlider::sub-page:horizontal {{
    background: {c.ACCENT};
    border-radius: {c.RADIUS};
}}

/* QRangeSlider (superqt) */
QRangeSlider {{
    qproperty-barColor: {c.ACCENT};
}}

QRangeSlider::groove:horizontal {{
    border: 1px solid {c.BORDER};
    height: 4px;
    background: {c.BG_ELEVATED};
    margin: 2px 0;
    border-radius: {c.RADIUS};
}}

QRangeSlider::handle:horizontal {{
    background: {c.TEXT_SECONDARY};
    border: 1px solid {c.TEXT_SECONDARY};
    width: 12px;
    height: 14px;
    margin: -6px 0;
    border-radius: {c.RADIUS};
}}

QRangeSlider::handle:horizontal:hover {{
    background: {c.TEXT_PRIMARY};
    border-color: {c.TEXT_PRIMARY};
}}

/* ProgressBar */
QProgressBar {{
    background-color: {c.BG_ELEVATED};
    border: 1px solid {c.BORDER};
    border-radius: {c.RADIUS};
    text-align: center;
    color: {c.TEXT_PRIMARY};
}}

QProgressBar::chunk {{
    background-color: {c.ACCENT};
    border-radius: {c.RADIUS};
}}

/* Menu Bar */
QMenuBar {{
    background-color: {c.BG_SURFACE};
    color: {c.TEXT_PRIMARY};
    border-bottom: 1px solid {c.BORDER};
}}

QMenuBar::item {{
    spacing: 3px;
    padding: 4px 8px;
    background: transparent;
    border-radius: {c.RADIUS};
}}

QMenuBar::item:selected {{
    background-color: {c.BG_ELEVATED_HOVER};
}}

QMenu {{
    background-color: {c.BG_SURFACE};
    color: {c.TEXT_PRIMARY};
    border: 1px solid {c.BORDER};
}}

QMenu::item {{
    padding: 4px 24px 4px 8px;
}}

QMenu::item:selected {{
    background-color: {c.ACCENT};
    color: {c.ON_ACCENT};
}}

QMenu::item:disabled {{
    color: {c.TEXT_DISABLED};
}}

QMenu::separator {{
    height: 1px;
    background: {c.BORDER};
    margin: 4px 6px;
}}

/* Where a horizontal and a vertical scroll bar meet; native (light) otherwise. */
QAbstractScrollArea::corner {{
    background: {c.BG_BASE};
    border: none;
}}

/* Scrollbars — slim rectangular handle, not a rounded pill. */
QScrollBar:vertical {{
    border: none;
    background: {c.BG_BASE};
    width: 11px;
    margin: 0;
}}

QScrollBar::handle:vertical {{
    background: {c.BORDER};
    min-height: 24px;
    border-radius: {c.RADIUS};
}}

QScrollBar::handle:vertical:hover {{
    background: {c.TEXT_SECONDARY};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: none;
}}

QScrollBar:horizontal {{
    border: none;
    background: {c.BG_BASE};
    height: 11px;
    margin: 0;
}}

QScrollBar::handle:horizontal {{
    background: {c.BORDER};
    min-width: 24px;
    border-radius: {c.RADIUS};
}}

QScrollBar::handle:horizontal:hover {{
    background: {c.TEXT_SECONDARY};
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
}}

QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
    background: none;
}}

/* Labels */
QLabel {{
    color: {c.TEXT_SECONDARY};
    background: transparent;
}}

QLabel:disabled {{
    color: {c.TEXT_DISABLED};
}}

/* Tab Widget */
QTabWidget::pane {{
    border: 1px solid {c.BORDER};
    border-radius: {c.RADIUS};
    background-color: {c.BG_BASE};
    top: -1px;
}}

QTabBar::tab {{
    background-color: {c.BG_SURFACE};
    color: {c.TEXT_SECONDARY};
    border: 1px solid {c.BORDER};
    border-bottom: none;
    padding: 5px 12px;
    border-top-left-radius: {c.RADIUS};
    border-top-right-radius: {c.RADIUS};
    margin-right: 2px;
}}

QTabBar::tab:selected {{
    background-color: {c.BG_BASE};
    color: {c.TEXT_PRIMARY};
}}

QTabBar::tab:hover:!selected {{
    background-color: {c.BG_ELEVATED_HOVER};
    color: {c.TEXT_PRIMARY};
}}

/* GroupBox */
QGroupBox {{
    border: 1px solid {c.BORDER};
    border-radius: {c.RADIUS};
    margin-top: 1em;
    padding-top: 10px;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top center;
    padding: 0 3px;
    color: {c.TEXT_SECONDARY};
}}

/* Dialogs & message boxes */
QDialog {{
    background-color: {c.BG_BASE};
    color: {c.TEXT_PRIMARY};
}}

QMessageBox {{
    background-color: {c.BG_BASE};
}}

QMessageBox QLabel {{
    color: {c.TEXT_PRIMARY};
}}

/* Toolbars — including the floating/dockable QMainWindow toolbar; left
   unstyled these fall back to native (light gray on most platforms)
   chrome, same as the other gaps above. */
QToolBar {{
    background-color: {c.BG_SURFACE};
    border: none;
    spacing: 3px;
    padding: 2px;
}}

QToolBar::separator {{
    background-color: {c.BORDER};
    width: 1px;
    margin: 4px 4px;
}}

QToolBar::handle {{
    background-color: {c.BORDER};
    width: 6px;
    margin: 4px 2px;
}}

/* Dock widgets */
QDockWidget {{
    color: {c.TEXT_PRIMARY};
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
}}

QDockWidget::title {{
    background-color: {c.BG_SURFACE};
    border: 1px solid {c.BORDER};
    padding: 5px 8px;
}}

/* Status bar */
QStatusBar {{
    background-color: {c.BG_SURFACE};
    color: {c.TEXT_SECONDARY};
    border-top: 1px solid {c.BORDER};
}}

QStatusBar::item {{
    border: none;
}}

/* Splitters */
QSplitter::handle {{
    background-color: {c.BORDER};
}}

QSplitter::handle:hover {{
    background-color: {c.ACCENT};
}}

QSplitter::handle:horizontal {{
    width: 2px;
}}

QSplitter::handle:vertical {{
    height: 2px;
}}
"""


class _NativeChromeFix(QObject):
    """Corrections for native chrome the stylesheet can't reach, applied to
    each widget as it is polished.

    Check and radio buttons are laid out by their full widget rect. The
    macOS style reports a check box's *layout* rect a few pixels
    shorter than the widget (it leaves out room for the native focus
    ring), so layouts give it less height than it paints and place it
    overlapping its neighbours. The stylesheet above repaints the
    indicator to fill the widget, so with it the reported rect is simply
    wrong: rows overlapped and clipped by a couple of pixels.
    `WA_LayoutUsesWidgetRect` opts a widget out of that adjustment.

    Tab bars don't draw their "base": in document mode the macOS style
    paints it as a light strip across the rest of the bar (beside the
    workspace's image tabs, for one), and no stylesheet rule reaches it.
    The tabs' own borders already separate them from the page."""

    def eventFilter(self, obj, event):  # noqa: N802
        if event.type() == QEvent.Type.Polish:
            if isinstance(obj, (QCheckBox, QRadioButton)):
                if not obj.testAttribute(Qt.WidgetAttribute.WA_LayoutUsesWidgetRect):
                    obj.setAttribute(Qt.WidgetAttribute.WA_LayoutUsesWidgetRect, True)
                    obj.updateGeometry()
            elif isinstance(obj, QTabBar) and obj.drawBase():
                obj.setDrawBase(False)
        return False


_chrome_fix = None


def apply(app) -> None:
    """Install pyvistra's look on `app`: the stylesheet, and the native
    chrome corrections it needs (`_NativeChromeFix`). Idempotent -- the one
    call every entry point makes instead of setting the stylesheet itself."""
    global _chrome_fix
    app.setStyleSheet(DARK_THEME)
    if _chrome_fix is None:
        _chrome_fix = _NativeChromeFix()
        app.installEventFilter(_chrome_fix)
