"""GUI entry point.

Logging handlers and log level are configured only here. Filter/service
modules obtain a logger but never configure handlers.
"""

from __future__ import annotations

import io
import logging
import os
import sys


def hide_console() -> None:
    """Hide the console window on Windows.

    The GUI is built as a console app so native libraries (torch/OpenMP) get
    valid std handles and load correctly; the console itself is hidden here so
    the user only sees the window.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes

        hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 0)  # SW_HIDE
    except Exception:  # noqa: BLE001
        pass


def repair_std_streams() -> None:
    """Give the process valid stdio handles.

    A windowed (``--noconsole``) frozen build has no console, so stdin/out/err
    are invalid. Native libraries such as PyTorch/OpenMP can then fail to
    initialize (``c10.dll`` WinError 1114). Point the handles at devnull.
    """
    if sys.stdout is not None and sys.stderr is not None:
        return
    try:
        fd = os.open(os.devnull, os.O_RDWR)
        for target in (0, 1, 2):
            try:
                os.dup2(fd, target)
            except OSError:
                pass
    except OSError:
        pass
    for name, mode in (("stdin", "r"), ("stdout", "w"), ("stderr", "w")):
        if getattr(sys, name, None) is None:
            try:
                setattr(sys, name, io.TextIOWrapper(open(os.devnull, mode + "b")))
            except OSError:
                pass


def configure_logging(level: int = logging.INFO) -> None:
    """Configure application-wide logging."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _selftest(path: str) -> int:
    """Load native deps under the current runtime and write the result.

    Used to verify a frozen build: run the exe with HPC_SELFTEST=<file>.
    """
    try:
        import torch
        from ultralytics import YOLO  # noqa: F401

        msg = f"OK torch={torch.__version__}"
    except Exception as exc:  # noqa: BLE001
        msg = f"FAIL {exc!r}"
    try:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(msg)
    except OSError:
        pass
    return 0


def main() -> int:
    """Run the GUI application."""
    selftest = os.environ.get("HPC_SELFTEST")
    repair_std_streams()
    configure_logging()

    if selftest:
        return _selftest(selftest)

    hide_console()

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
