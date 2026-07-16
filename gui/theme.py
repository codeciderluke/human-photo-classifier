"""Application dark theme.

Provides a modern dark color palette and a Qt stylesheet (QSS), keeping
styling separate from the GUI widget code.
"""

from __future__ import annotations

# Color palette (dark)
COLORS = {
    "bg": "#0f1115",          # base background
    "surface": "#171a21",     # card/panel surface
    "surface_alt": "#1e222b", # input widget surface
    "border": "#2a2f3a",      # border
    "border_focus": "#3d8bfd",
    "text": "#e6e9ef",        # primary text
    "text_muted": "#8b93a7",  # secondary text
    "accent": "#3d8bfd",      # accent (blue)
    "accent_hover": "#5c9dff",
    "accent_pressed": "#2f74d0",
    "success": "#3fb950",
    "danger": "#f85149",
    "danger_hover": "#ff6259",
    "track": "#22262f",       # progress track
}


def build_stylesheet() -> str:
    """Build the global QSS string for the app."""
    c = COLORS
    return f"""
    * {{
        font-family: "Segoe UI", "Malgun Gothic", sans-serif;
        font-size: 15px;
        color: {c['text']};
        outline: none;
    }}

    QWidget#root {{
        background-color: {c['bg']};
    }}

    /* card container */
    QFrame#card {{
        background-color: {c['surface']};
        border: 1px solid {c['border']};
        border-radius: 16px;
    }}

    QLabel#title {{
        font-size: 30px;
        font-weight: 700;
        color: {c['text']};
    }}

    QLabel#subtitle {{
        font-size: 15px;
        color: {c['text_muted']};
    }}

    QLabel#footer {{
        font-size: 12px;
        color: {c['text_muted']};
    }}

    QLabel#sectionLabel {{
        font-size: 13px;
        font-weight: 600;
        color: {c['text_muted']};
        letter-spacing: 1px;
    }}

    QLabel#fieldLabel {{
        color: {c['text_muted']};
        font-size: 14px;
    }}

    /* input widgets */
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
        background-color: {c['surface_alt']};
        border: 1px solid {c['border']};
        border-radius: 10px;
        padding: 11px 14px;
        selection-background-color: {c['accent']};
        min-height: 20px;
    }}

    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
        border: 1px solid {c['border_focus']};
    }}

    QComboBox::drop-down {{
        border: none;
        width: 22px;
    }}

    QComboBox QAbstractItemView {{
        background-color: {c['surface_alt']};
        border: 1px solid {c['border']};
        border-radius: 8px;
        selection-background-color: {c['accent']};
        padding: 4px;
    }}

    QSpinBox::up-button, QSpinBox::down-button,
    QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
        width: 16px;
        background: transparent;
        border: none;
    }}

    /* buttons */
    QPushButton {{
        background-color: {c['surface_alt']};
        border: 1px solid {c['border']};
        border-radius: 10px;
        padding: 11px 18px;
        color: {c['text']};
    }}

    QPushButton:hover {{
        border: 1px solid {c['border_focus']};
    }}

    QPushButton#primary {{
        background-color: {c['accent']};
        border: none;
        font-weight: 600;
        font-size: 16px;
        padding: 14px 26px;
    }}

    QPushButton#danger {{
        font-size: 16px;
    }}

    QPushButton#primary:hover {{
        background-color: {c['accent_hover']};
    }}

    QPushButton#primary:pressed {{
        background-color: {c['accent_pressed']};
    }}

    QPushButton#danger {{
        background-color: {c['danger']};
        border: none;
        font-weight: 600;
        padding: 10px 18px;
    }}

    QPushButton#danger:hover {{
        background-color: {c['danger_hover']};
    }}

    QPushButton:disabled {{
        background-color: {c['surface_alt']};
        color: {c['text_muted']};
        border: 1px solid {c['border']};
    }}

    /* progress bar */
    QProgressBar {{
        background-color: {c['track']};
        border: none;
        border-radius: 7px;
        height: 14px;
        text-align: center;
        color: transparent;
    }}

    QProgressBar::chunk {{
        background-color: {c['accent']};
        border-radius: 7px;
    }}

    /* log area */
    QPlainTextEdit#log {{
        background-color: {c['bg']};
        border: 1px solid {c['border']};
        border-radius: 12px;
        padding: 14px;
        font-family: "Cascadia Mono", "Consolas", monospace;
        font-size: 14px;
        color: {c['text_muted']};
    }}

    /* scrollbar */
    QScrollBar:vertical {{
        background: transparent;
        width: 10px;
        margin: 2px;
    }}
    QScrollBar::handle:vertical {{
        background: {c['border']};
        border-radius: 5px;
        min-height: 30px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {c['text_muted']};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}

    /* checkbox */
    QCheckBox {{
        spacing: 8px;
    }}
    QCheckBox::indicator {{
        width: 18px;
        height: 18px;
        border-radius: 5px;
        border: 1px solid {c['border']};
        background: {c['surface_alt']};
    }}
    QCheckBox::indicator:checked {{
        background: {c['accent']};
        border: 1px solid {c['accent']};
    }}
    """
