"""PyQt5 GUI package.

The GUI layer only handles the user interface: folder selection, settings
input, progress display, log display, and stopping work. Actual detection is
handled by the ``filters`` package, and file handling by the ``services``
package.
"""

from .main_window import MainWindow

__all__ = ["MainWindow"]
