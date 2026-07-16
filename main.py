"""Application entry point.

Logging handlers and log level are configured only here. Filter/service
modules obtain a logger but never configure handlers.
"""

from __future__ import annotations

import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    """Configure application-wide logging."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> int:
    """Run the GUI application."""
    configure_logging()

    from pathlib import Path

    from PyQt5.QtGui import QIcon
    from PyQt5.QtWidgets import QApplication

    from gui import MainWindow

    app = QApplication(sys.argv)
    icon_path = Path(__file__).resolve().parent / "assets" / "icon.ico"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
