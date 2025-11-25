DARK_THEME = """
/* --- General --- */
QMainWindow, QDialog, QWidget {
    background-color: #2b2b2b;
    color: #e0e0e0;
    font-family: sans-serif;
    font-size: 12pt;
}

/* --- Labels --- */
QLabel {
    color: #e0e0e0;
}
QLabel#infoLabel {
    background-color: #202020;
    color: #40a9ff;
    padding: 1px 5px; 
    font-family: "Consolas", "Monospace";
    border-top: 1px solid #3a3a3a;
    font-size: 9pt;
}
QLabel#titleLabel {
    font-size: 12pt;
    font-weight: bold;
    color: #ffffff;
    padding-bottom: 5px;
}

/* --- Standard Sliders --- */
QSlider::groove:horizontal {
    border: 1px solid #3a3a3a;
    height: 6px;
    background: #3a3a3a;
    margin: 2px 0;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: #40a9ff;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #e0e0e0;
    border: 1px solid #e0e0e0;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QSlider::handle:horizontal:hover {
    background: #ffffff;
    border-color: #ffffff;
}

/* --- QRangeSlider (SuperQt) --- */
/* Range slider uses 'window' background for the groove often, so we enforce it */
QRangeSlider {
    qproperty-barColor: #40a9ff;
}
QRangeSlider::groove:horizontal {
    border: 1px solid #3a3a3a;
    height: 6px;
    background: #3a3a3a;
    margin: 2px 0;
    border-radius: 3px;
}

/* --- Buttons --- */
QPushButton {
    background-color: #3a3a3a;
    border: 1px solid #555;
    border-radius: 4px;
    padding: 6px 12px;
}
QPushButton:hover {
    background-color: #454545;
    border-color: #40a9ff;
}
QPushButton:pressed {
    background-color: #40a9ff;
    color: #111;
}

/* --- Combo Boxes --- */
QComboBox {
    background-color: #3a3a3a;
    border: 1px solid #555;
    border-radius: 4px;
    padding: 4px 10px;
    min-width: 6em;
}
QComboBox:hover {
    border-color: #40a9ff;
}
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 20px;
    border-left-width: 0px;
    border-top-right-radius: 4px;
    border-bottom-right-radius: 4px;
}
QComboBox QAbstractItemView {
    background-color: #3a3a3a;
    border: 1px solid #555;
    selection-background-color: #40a9ff;
    selection-color: #ffffff;
}

/* --- Menus --- */
QMenuBar {
    background-color: #2b2b2b;
    border-bottom: 1px solid #3a3a3a;
}
QMenuBar::item {
    padding: 6px 10px;
    background: transparent;
}
QMenuBar::item:selected {
    background: #3a3a3a;
}
QMenu {
    background-color: #2b2b2b;
    border: 1px solid #555;
}
QMenu::item {
    padding: 6px 24px;
}
QMenu::item:selected {
    background-color: #40a9ff;
    color: white;
}
"""
